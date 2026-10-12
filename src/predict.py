"""Run the tuned model on one invoice text.

  python -m src.predict --adapter outputs/qwen0.5b-lora --text "Invoice 4471 from Acme Ltd ..."
  python -m src.predict --adapter outputs/qwen0.5b-lora --file sample_invoice.txt
"""
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from src.common import build_messages, parse_json_output
from src.train import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="omit to run the untuned base model")
    ap.add_argument("--text")
    ap.add_argument("--file")
    a = ap.parse_args()
    if a.text:
        text = a.text
    else:
        with open(a.file, encoding="utf-8") as f:
            text = f.read()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model, torch.float32)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()
    prompt = tok.apply_chat_template(build_messages(text), add_generation_prompt=True, tokenize=False)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=160, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = parse_json_output(raw)
    print(json.dumps(parsed, indent=2) if parsed else f"(could not parse JSON)\n{raw}")


if __name__ == "__main__":
    main()
