"""Evaluate a base model (zero/few-shot) or base + LoRA adapter on a held-out split.

Examples
  python -m src.evaluate --name base-zeroshot --split data/test_ood.jsonl
  python -m src.evaluate --name base-fewshot  --split data/test_ood.jsonl --fewshot 2
  python -m src.evaluate --name tuned         --split data/test_ood.jsonl --adapter outputs/qwen0.5b-lora
"""
import argparse
import json
import os
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer

from src.common import build_messages, read_jsonl, write_jsonl
from src.metrics import aggregate, score_one
from src.train import load_model


def pick_shots(train_path: str, k: int) -> list[dict]:
    """Deterministic few-shot examples: the first training row of k different templates."""
    shots, seen = [], set()
    for row in read_jsonl(train_path):
        if row["template"] not in seen:
            seen.add(row["template"])
            shots.append(row)
        if len(shots) == k:
            break
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="label for this run, e.g. base-zeroshot")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="data/test_ood.jsonl")
    ap.add_argument("--train-data", default="data/train.jsonl")
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="evaluate only the first N rows (0 = all)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"  # decoder-only models must be left-padded for batched generation
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = load_model(a.model, dtype)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()  # fold LoRA into weights
    model = model.to(device).eval()

    rows = read_jsonl(a.split)
    rows = rows[:a.n] if a.n else rows
    shots = pick_shots(a.train_data, a.fewshot) if a.fewshot else []
    prompts = [tok.apply_chat_template(build_messages(r["text"], shots), add_generation_prompt=True,
                                       tokenize=False) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))  # similar lengths -> less padding
    raw = [None] * len(rows)
    t0 = time.time()
    for b in range(0, len(order), a.batch_size):
        idx = order[b:b + a.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=pad_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts, strict=True):
            raw[i] = t
        print(f"[{a.name}] {min(b + a.batch_size, len(order))}/{len(order)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    elapsed = time.time() - t0

    scores = [score_one(raw[i], rows[i]["target"]) for i in range(len(rows))]
    by_template = defaultdict(list)
    for r, s in zip(rows, scores, strict=True):
        by_template[r["template"]].append(s)

    metrics = {
        "name": a.name, "model": a.model, "adapter": a.adapter, "split": os.path.basename(a.split),
        "fewshot": a.fewshot, "device": device, "seconds_per_sample": round(elapsed / len(rows), 2),
        **aggregate(scores),
        "by_template": {t: aggregate(s) for t, s in by_template.items()},
    }
    os.makedirs(a.out, exist_ok=True)
    split_tag = os.path.splitext(os.path.basename(a.split))[0]
    stem = os.path.join(a.out, f"{a.name}.{split_tag}")
    with open(stem + ".metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    write_jsonl(stem + ".predictions.jsonl", [
        {"id": r["id"], "template": r["template"], "gold": r["target"], "raw_output": raw[i],
         "exact_match": scores[i]["exact_match"]} for i, r in enumerate(rows)])
    print(f"\n{a.name} on {split_tag}: exact_match={metrics['exact_match']:.3f} "
          f"field_acc={metrics['field_accuracy_macro']:.3f} json_valid={metrics['json_valid']:.3f} "
          f"({metrics['seconds_per_sample']}s/sample)")


if __name__ == "__main__":
    main()
