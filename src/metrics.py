"""Scoring for extraction outputs. Pure Python so it is cheap to unit test and run in CI."""
from collections import defaultdict

from src.common import FIELDS, parse_json_output


def _as_float(x):
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def field_correct(field: str, pred, gold) -> bool:
    if field == "total":
        p, g = _as_float(pred), _as_float(gold)
        return p is not None and g is not None and abs(p - g) < 0.005
    if gold is None:
        return pred is None
    return isinstance(pred, str) and pred.strip() == str(gold).strip()


def schema_valid(obj) -> bool:
    """Strict: exactly the expected keys, with the expected JSON types."""
    if not isinstance(obj, dict) or set(obj) != set(FIELDS):
        return False
    if isinstance(obj["total"], bool) or not isinstance(obj["total"], (int, float)):
        return False
    if obj["po_number"] is not None and not isinstance(obj["po_number"], str):
        return False
    return all(isinstance(obj[k], str) for k in FIELDS if k not in ("total", "po_number"))


def score_one(raw_output: str, gold: dict) -> dict:
    obj = parse_json_output(raw_output)
    per_field = {f: bool(obj) and field_correct(f, obj.get(f), gold[f]) for f in FIELDS}
    return {
        "json_valid": obj is not None,
        "schema_valid": schema_valid(obj),
        "fields": per_field,
        "exact_match": all(per_field.values()),
    }


def aggregate(scores: list[dict]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    per_field = defaultdict(int)
    for s in scores:
        for f, ok in s["fields"].items():
            per_field[f] += ok
    field_acc = {f: per_field[f] / n for f in FIELDS}
    return {
        "n": n,
        "json_valid": sum(s["json_valid"] for s in scores) / n,
        "schema_valid": sum(s["schema_valid"] for s in scores) / n,
        "exact_match": sum(s["exact_match"] for s in scores) / n,
        "field_accuracy_macro": sum(field_acc.values()) / len(FIELDS),
        "field_accuracy": field_acc,
    }
