"""Task definition shared by data generation, training, evaluation and inference."""
import json

FIELDS = ["invoice_number", "vendor", "invoice_date", "due_date", "currency", "total", "po_number"]

SYSTEM_PROMPT = (
    "You extract invoice fields from text. Reply with a single JSON object and nothing else, "
    "with exactly these keys: invoice_number (string), vendor (string, the company issuing the invoice), "
    "invoice_date (YYYY-MM-DD), due_date (YYYY-MM-DD), currency (ISO 4217 code), "
    "total (number, the final amount due including tax), po_number (string or null)."
)


def serialize_target(target: dict) -> str:
    """Canonical compact JSON with fixed key order (fewer tokens to generate, easier to learn)."""
    return json.dumps({k: target[k] for k in FIELDS}, separators=(",", ":"), ensure_ascii=False)


def build_messages(text: str, shots: list[dict] | None = None) -> list[dict]:
    """Chat messages for one extraction request. `shots` adds few-shot example turns (baseline only)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in shots or []:
        messages.append({"role": "user", "content": shot["text"]})
        messages.append({"role": "assistant", "content": serialize_target(shot["target"])})
    messages.append({"role": "user", "content": text})
    return messages


def parse_json_output(raw: str) -> dict | None:
    """Pull the first JSON object out of a model reply (models sometimes add prose or code fences)."""
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(raw[start:])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            start = raw.find("{", start + 1)
    return None


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
