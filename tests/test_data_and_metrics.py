import json

from src import generate_data as gd
from src.common import FIELDS, parse_json_output, serialize_target
from src.metrics import aggregate, field_correct, schema_valid, score_one

GOLD = {"invoice_number": "INV-1", "vendor": "Acme LLC", "invoice_date": "2025-01-02",
        "due_date": "2025-02-01", "currency": "USD", "total": 1234.5, "po_number": None}


def test_generation_is_deterministic():
    a = gd.build_split("x", 20, gd.TRAIN_TEMPLATES, seed=1)
    b = gd.build_split("x", 20, gd.TRAIN_TEMPLATES, seed=1)
    assert a == b


def test_targets_have_all_fields_and_consistent_dates():
    for row in gd.build_split("x", 60, gd.TRAIN_TEMPLATES + gd.OOD_TEMPLATES, seed=3):
        assert list(row["target"]) == FIELDS
        assert row["target"]["invoice_date"] < row["target"]["due_date"]
        assert row["target"]["currency"] in {"USD", "EUR", "GBP", "CAD"}


def test_answer_is_recoverable_from_text():
    """The vendor and invoice number must literally appear in the text (label typos never hit values)."""
    for row in gd.build_split("x", 80, gd.TRAIN_TEMPLATES + gd.OOD_TEMPLATES, seed=5):
        text = row["text"].lower()
        assert row["target"]["vendor"].lower() in text
        assert row["target"]["invoice_number"].lower() in text


def test_ood_templates_are_disjoint_from_train_templates():
    train = {t.__name__ for t in gd.TRAIN_TEMPLATES}
    ood = {t.__name__ for t in gd.OOD_TEMPLATES}
    assert not train & ood


def test_no_text_shared_across_splits():
    train = {r["text"] for r in gd.build_split("train", 50, gd.TRAIN_TEMPLATES, 7)}
    ood = {r["text"] for r in gd.build_split("ood", 50, gd.OOD_TEMPLATES, 10)}
    assert not train & ood


def test_number_formats_roundtrip_values():
    assert gd.fmt_number(1234.5, "us") == "1,234.50"
    assert gd.fmt_number(1234.5, "eu") == "1.234,50"
    assert gd.fmt_number(1234.5, "space") == "1 234,50"


def test_parse_json_output_handles_noise():
    assert parse_json_output('Sure! ```json\n{"a": 1}\n``` done') == {"a": 1}
    assert parse_json_output("no json here") is None
    assert parse_json_output("{broken") is None
    assert parse_json_output("[1, 2]") is None


def test_serialize_target_is_canonical():
    shuffled = dict(reversed(list(GOLD.items())))
    assert serialize_target(shuffled) == serialize_target(GOLD)
    assert json.loads(serialize_target(GOLD)) == GOLD


def test_perfect_prediction_scores_exact_match():
    s = score_one(serialize_target(GOLD), GOLD)
    assert s["exact_match"] and s["schema_valid"] and s["json_valid"]


def test_total_tolerates_numeric_strings_but_schema_is_strict():
    pred = {**GOLD, "total": "1234.50"}
    s = score_one(json.dumps(pred), GOLD)
    assert s["fields"]["total"] is True  # value is right...
    assert s["schema_valid"] is False  # ...but the type is wrong


def test_field_level_scoring_and_null_handling():
    assert field_correct("po_number", None, None)
    assert not field_correct("po_number", "PO-1", None)
    assert not field_correct("vendor", "acme llc", "Acme LLC")  # exact string match
    s = score_one(json.dumps({**GOLD, "currency": "EUR"}), GOLD)
    assert not s["exact_match"] and s["fields"]["vendor"] and not s["fields"]["currency"]


def test_invalid_output_scores_zero():
    s = score_one("I cannot help with that", GOLD)
    assert not s["json_valid"] and not any(s["fields"].values())


def test_schema_valid_rejects_extra_and_missing_keys():
    assert schema_valid(GOLD)
    assert not schema_valid({**GOLD, "extra": 1})
    assert not schema_valid({k: v for k, v in GOLD.items() if k != "vendor"})


def test_aggregate():
    good = score_one(serialize_target(GOLD), GOLD)
    bad = score_one("nope", GOLD)
    agg = aggregate([good, bad])
    assert agg["n"] == 2 and agg["exact_match"] == 0.5 and agg["json_valid"] == 0.5
