"""LoRA fine-tuning of a causal LM for invoice -> JSON extraction.

A plain PyTorch loop (no Trainer) so every step is explicit:
  1. render each example with the model's chat template
  2. mask the prompt tokens so loss is computed ONLY on the assistant's JSON answer
  3. train low-rank adapters (LoRA) on all attention + MLP projections; base weights stay frozen
  4. cosine LR schedule with warmup, gradient accumulation, gradient clipping
  5. evaluate validation loss each epoch and save only the small adapter
Runs on CPU (slow, fine for a 0.5B model) or a single GPU (bf16 when supported).
"""
import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.common import build_messages, read_jsonl, serialize_target

IGNORE = -100
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model(name: str, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:  # older transformers used `torch_dtype`
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


def encode(example: dict, tok, max_len: int):
    """-> (input_ids, labels) with prompt tokens masked, or None if it does not fit max_len."""
    messages = build_messages(example["text"])
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": serialize_target(example["target"])}],
        tokenize=False,
    )
    assert full.startswith(prompt), "chat template must render the prompt as a prefix of the full text"
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_len:
        return None
    labels = [IGNORE] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


def collate(batch, pad_id: int):
    longest = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = longest - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [IGNORE] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


@torch.no_grad()
def eval_loss(model, data, pad_id, device, bs=8):
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(data), bs):
        ids, labels, attn = (t.to(device) for t in collate(data[i:i + bs], pad_id))
        logits = model(input_ids=ids, attention_mask=attn).logits
        shift_logits, shift_labels = logits[:, :-1], labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(), shift_labels.reshape(-1),
            ignore_index=IGNORE, reduction="sum")
        total += loss.item()
        count += (shift_labels != IGNORE).sum().item()
    model.train()
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="outputs/qwen0.5b-lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--max-train", type=int, default=0, help="use only the first N training rows (0 = all)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-checkpointing", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32)
    print(f"device={device} dtype={dtype} threads={torch.get_num_threads()}")

    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def prep(path, limit=0):
        rows = read_jsonl(path)
        rows = rows[:limit] if limit else rows
        enc = [encode(r, tok, a.max_len) for r in rows]
        kept = [e for e in enc if e is not None]
        lens = sorted(len(ids) for ids, _ in kept)
        print(f"{path}: {len(kept)}/{len(rows)} fit max_len={a.max_len}; "
              f"tokens median={lens[len(lens) // 2]} max={lens[-1]}")
        return kept

    train, val = prep(a.train, a.max_train), prep(a.val)

    model = load_model(a.model, dtype).to(device)
    if a.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                      target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)")
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(train) / (a.batch_size * a.grad_accum))
    total_steps = a.max_steps or math.ceil(steps_per_epoch * a.epochs)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, int(0.05 * total_steps)),
                                            num_training_steps=total_steps)
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "w", encoding="utf-8")
    best = eval_loss(model, val, pad_id, device)
    print(f"val loss before training: {best:.4f}")
    log.write(json.dumps({"event": "eval", "step": 0, "val_loss": best}) + "\n")

    step, t0, run_loss, run_n = 0, time.time(), 0.0, 0
    epoch = 0
    while step < total_steps:
        order = list(range(len(train)))
        random.shuffle(order)
        micro = [order[i:i + a.batch_size] for i in range(0, len(order), a.batch_size)]
        for m_idx, idxs in enumerate(micro):
            ids, labels, attn = (t.to(device) for t in collate([train[i] for i in idxs], pad_id))
            logits = model(input_ids=ids, attention_mask=attn).logits
            sl, tl = logits[:, :-1], labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.size(-1)).float(), tl.reshape(-1), ignore_index=IGNORE)
            (loss / a.grad_accum).backward()
            run_loss += loss.item()
            run_n += 1
            if (m_idx + 1) % a.grad_accum == 0 or m_idx == len(micro) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or step == 1:
                    rec = {"event": "train", "step": step, "loss": run_loss / run_n,
                           "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                    print(f"step {step}/{total_steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                          f"elapsed {rec['elapsed_s']}s", flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    run_loss, run_n = 0.0, 0
                if step >= total_steps:
                    break
        epoch += 1
        vl = eval_loss(model, val, pad_id, device)
        print(f"== epoch {epoch} val loss {vl:.4f} (best {best:.4f})")
        log.write(json.dumps({"event": "eval", "step": step, "val_loss": vl}) + "\n")
        log.flush()

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "train_config.json"), "w") as f:
        json.dump({**vars(a), "device": device, "trainable_params": n_train, "total_params": n_total,
                   "train_seconds": round(time.time() - t0, 1), "steps": step}, f, indent=2)
    print(f"saved adapter to {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
