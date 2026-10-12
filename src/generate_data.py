"""Synthetic invoice text -> structured JSON dataset.

Everything is fictional and generated from a seed: company names come from word lists, phone numbers
use the reserved 555-01xx range, and no payment-card, bank-account or personal data is produced.

Splits are designed to test generalization, not memorization:
  train / val   : templates TRAIN_TEMPLATES
  test_id       : fresh records rendered with the same templates
  test_ood      : fresh records rendered with templates NEVER seen in training (different layout,
                  date format and number format) -- the honest measure of whether tuning generalizes.
"""
import argparse
import random
from datetime import date, timedelta

from src.common import write_jsonl

PREFIX = ["Northwind", "Blue Harbor", "Summit", "Ironwood", "Crescent", "Alder", "Pinecrest", "Redline",
          "Brightwater", "Kestrel", "Maple Ridge", "Oakmont", "Silverline", "Harborview", "Granite",
          "Lumen", "Ashford", "Copperfield", "Willowbrook", "Stonebridge", "Evergreen", "Falcon"]
NOUN = ["Logistics", "Supply", "Analytics", "Packaging", "Industrial", "Software", "Staffing", "Foods",
        "Electric", "Printing", "Maintenance", "Freight", "Office Products", "Consulting", "Textiles"]
SUFFIX = ["LLC", "Inc.", "Ltd", "GmbH", "Co.", "Corp.", "Group", "& Sons", "PLC"]
ITEM_WORDS = ["Widget", "Service fee", "Consulting hours", "Freight", "Licence renewal", "Pallet wrap",
              "Toner cartridge", "Maintenance plan", "Cable assembly", "Onsite support", "Storage fee"]
FIRST = ["Dana", "Priya", "Marcus", "Elena", "Tom", "Aisha", "Jon", "Mei", "Carlos", "Nina"]

CURRENCIES = ["USD", "USD", "USD", "EUR", "GBP", "CAD"]
MONTHS_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]


# ------------------------------------------------------------------ record

def company(rng):
    return f"{rng.choice(PREFIX)} {rng.choice(NOUN)} {rng.choice(SUFFIX)}"


def invoice_number(rng):
    return rng.choice([
        lambda: f"INV-{rng.randrange(2023, 2026)}-{rng.randrange(1, 9999):05d}",
        lambda: str(rng.randrange(10000, 999999)),
        lambda: f"{rng.choice('ABCDEFGH')}-{rng.randrange(1000, 99999)}",
        lambda: f"{rng.randrange(2023, 2026)}/{rng.randrange(1, 999):04d}",
        lambda: f"SI{rng.randrange(100000, 999999)}",
    ])()


def make_record(rng: random.Random) -> dict:
    issue = date(2024, 1, 1) + timedelta(days=rng.randrange(0, 730))
    terms = rng.choice([14, 15, 30, 30, 45, 60])
    n_items = rng.randrange(1, 5)
    items = []
    for _ in range(n_items):
        qty = rng.randrange(1, 40)
        unit = round(rng.uniform(4, 900), 2)
        items.append({"desc": rng.choice(ITEM_WORDS), "qty": qty, "unit": unit,
                      "amount": round(qty * unit, 2)})
    subtotal = round(sum(i["amount"] for i in items), 2)
    tax = round(subtotal * rng.choice([0, 0, 0.05, 0.08, 0.0825, 0.2]), 2)
    po = f"PO-{rng.randrange(10000, 99999)}" if rng.random() < 0.55 else None
    return {
        "vendor": company(rng),
        "customer": company(rng),
        "invoice_number": invoice_number(rng),
        "invoice_date": issue,
        "due_date": issue + timedelta(days=terms),
        "terms": terms,
        "currency": rng.choice(CURRENCIES),
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": round(subtotal + tax, 2),
        "po_number": po,
        "contact": rng.choice(FIRST),
        "phone": f"555-01{rng.randrange(0, 100):02d}",
    }


def target_of(rec: dict) -> dict:
    return {
        "invoice_number": rec["invoice_number"],
        "vendor": rec["vendor"],
        "invoice_date": rec["invoice_date"].isoformat(),
        "due_date": rec["due_date"].isoformat(),
        "currency": rec["currency"],
        "total": rec["total"],
        "po_number": rec["po_number"],
    }


# ------------------------------------------------------------------ formatting helpers

def fmt_date(d: date, style: str) -> str:
    return {
        "iso": d.isoformat(),
        "us": d.strftime("%m/%d/%Y"),
        "eu": d.strftime("%d.%m.%Y"),
        "long": f"{MONTHS_FULL[d.month - 1]} {d.day}, {d.year}",
        "dmy": f"{d.day} {MONTHS_ABBR[d.month - 1]} {d.year}",
        "short": f"{d.day:02d}-{MONTHS_ABBR[d.month - 1]}-{d.year % 100:02d}",
        "weekday": d.strftime("%A, ") + f"{d.day} {MONTHS_FULL[d.month - 1]} {d.year}",
    }[style]


def fmt_number(x: float, style: str) -> str:
    us = f"{x:,.2f}"
    if style == "us":
        return us
    if style == "eu":
        return us.replace(",", "X").replace(".", ",").replace("X", ".")
    if style == "space":
        return us.replace(",", " ").replace(".", ",")
    raise ValueError(style)


def fmt_money(x: float, cur: str, num_style: str, rng: random.Random, force_code=False) -> str:
    n = fmt_number(x, num_style)
    symbol = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "C$"}[cur]
    if force_code or rng.random() < 0.4:
        return f"{cur} {n}" if rng.random() < 0.5 else f"{n} {cur}"
    return f"{symbol}{n}" if rng.random() < 0.7 else f"{symbol} {n}"


def default_num_style(cur: str, rng: random.Random) -> str:
    if cur == "EUR":
        return rng.choice(["eu", "eu", "space"])
    return "us"


def ocr_typo(label: str, rng: random.Random, p=0.25) -> str:
    """Corrupt a *label* (never a value) the way OCR does, so the answer stays recoverable."""
    if rng.random() > p:
        return label
    swaps = {"I": "l", "l": "1", "o": "0", "O": "0", "a": "e", "t": "1"}
    idx = [i for i, ch in enumerate(label) if ch in swaps]
    if not idx:
        return label
    i = rng.choice(idx)
    return label[:i] + swaps[label[i]] + label[i + 1:]


def shuffled(rng, lines):
    lines = list(lines)
    rng.shuffle(lines)
    return lines


# ------------------------------------------------------------------ TRAIN templates

def tpl_keyvalue(r, rng):
    ds = rng.choice(["iso", "us", "long", "dmy", "eu"])
    ns = default_num_style(r["currency"], rng)
    lab = lambda opts: rng.choice(opts)  # noqa: E731
    lines = [
        f"{lab(['Invoice No', 'Invoice #', 'Inv. Number', 'Invoice Number'])}: {r['invoice_number']}",
        f"{lab(['Vendor', 'From', 'Supplier', 'Issued by'])}: {r['vendor']}",
        f"{lab(['Bill To', 'Customer', 'Sold To'])}: {r['customer']}",
        f"{lab(['Date', 'Invoice Date', 'Issued'])}: {fmt_date(r['invoice_date'], ds)}",
        f"{lab(['Due', 'Due Date', 'Payment Due'])}: {fmt_date(r['due_date'], ds)}",
        f"Subtotal: {fmt_money(r['subtotal'], r['currency'], ns, rng)}",
        f"Tax: {fmt_money(r['tax'], r['currency'], ns, rng)}",
        f"{lab(['Total Due', 'Total', 'Amount Due', 'Balance Due'])}: "
        f"{fmt_money(r['total'], r['currency'], ns, rng)}",
    ]
    if r["po_number"]:
        lines.append(f"{lab(['PO', 'PO Number', 'Purchase Order', 'Your Ref'])}: {r['po_number']}")
    return "\n".join(shuffled(rng, lines))


def tpl_email(r, rng):
    ds = rng.choice(["long", "dmy", "us", "iso"])
    ns = default_num_style(r["currency"], rng)
    total = fmt_money(r["total"], r["currency"], ns, rng)
    po = f" Please reference {r['po_number']} when paying." if r["po_number"] else ""
    return (
        f"Subject: Invoice {r['invoice_number']} from {r['vendor']}\n\n"
        f"Hi team,\n\nPlease find attached invoice {r['invoice_number']} from {r['vendor']} "
        f"dated {fmt_date(r['invoice_date'], ds)}, addressed to {r['customer']}. The amount of {total} "
        f"(tax included) is payable by {fmt_date(r['due_date'], ds)}.{po}\n\n"
        f"Questions? Call {r['contact']} on {r['phone']}.\n\nThanks,\n{r['contact']}"
    )


def tpl_ocr(r, rng):
    ds = rng.choice(["us", "us", "eu", "dmy"])
    ns = default_num_style(r["currency"], rng)
    t = lambda s: ocr_typo(s, rng)  # noqa: E731
    lines = [
        r["vendor"].upper(),
        f"TEL {r['phone']}   PAGE 1 OF 1",
        f"{t('INVOICE')} {t('NO.')} {r['invoice_number']}    {t('DATE')} {fmt_date(r['invoice_date'], ds)}",
        f"{t('BILL TO')}: {r['customer']}",
    ]
    if r["po_number"]:
        lines.append(f"{t('P.O.')} {r['po_number']}")
    for it in r["items"]:
        lines.append(f"{it['desc']:<18} {it['qty']:>3} x {fmt_number(it['unit'], ns):>10}  "
                     f"{fmt_number(it['amount'], ns):>11}")
    lines += [
        f"{t('SUBTOTAL')} {fmt_number(r['subtotal'], ns)}",
        f"{t('TAX')} {fmt_number(r['tax'], ns)}",
        f"{t('TOTAL')} {r['currency']} {fmt_number(r['total'], ns)}",
        f"{t('TERMS')}: NET {r['terms']}   {t('DUE')} {fmt_date(r['due_date'], ds)}",
    ]
    return "\n".join(lines)


def tpl_table(r, rng):
    ds = rng.choice(["iso", "long", "dmy"])
    ns = default_num_style(r["currency"], rng)
    cur = r["currency"]
    rows = "\n".join(
        f"| {it['desc']} | {it['qty']} | {fmt_money(it['unit'], cur, ns, rng, True)} | "
        f"{fmt_money(it['amount'], cur, ns, rng, True)} |" for it in r["items"])
    po = f"\nPurchase order: {r['po_number']}" if r["po_number"] else ""
    return (
        f"{r['vendor']}\nInvoice {r['invoice_number']}\nIssue date: {fmt_date(r['invoice_date'], ds)}"
        f"\nDue date: {fmt_date(r['due_date'], ds)}\nCustomer: {r['customer']}{po}\n\n"
        f"| Item | Qty | Unit | Amount |\n{rows}\n\n"
        f"Subtotal {fmt_money(r['subtotal'], cur, ns, rng, True)}\n"
        f"Tax {fmt_money(r['tax'], cur, ns, rng, True)}\n"
        f"Total {fmt_money(r['total'], cur, ns, rng, True)}"
    )


def tpl_remittance(r, rng):
    ds = rng.choice(["long", "us", "dmy"])
    ns = default_num_style(r["currency"], rng)
    po = f" under purchase order {r['po_number']}" if r["po_number"] else ""
    return (
        f"Remittance advice from {r['customer']}. We are settling invoice number {r['invoice_number']}, "
        f"issued by {r['vendor']} on {fmt_date(r['invoice_date'], ds)}{po}. Amount remitted: "
        f"{fmt_money(r['total'], r['currency'], ns, rng)}. Payment terms were net {r['terms']} days, "
        f"so it fell due on {fmt_date(r['due_date'], ds)}."
    )


TRAIN_TEMPLATES = [tpl_keyvalue, tpl_email, tpl_ocr, tpl_table, tpl_remittance]


# ------------------------------------------------------------------ OOD templates (test only)

def tpl_statement(r, rng):
    """Two-column statement layout; abbreviated 2-digit-year dates; ship-date distractor."""
    ns = default_num_style(r["currency"], rng)
    ship = r["invoice_date"] - timedelta(days=rng.randrange(1, 6))
    lines = [
        f"FROM      {r['vendor']}",
        f"TO        {r['customer']}",
        f"REF       {r['invoice_number']}",
        f"ISSUED    {fmt_date(r['invoice_date'], 'short')}",
        f"SHIPPED   {fmt_date(ship, 'short')}",
        f"PAYABLE BY {fmt_date(r['due_date'], 'short')}",
        f"NET       {fmt_money(r['subtotal'], r['currency'], ns, rng, True)}",
        f"VAT/TAX   {fmt_money(r['tax'], r['currency'], ns, rng, True)}",
        f"AMOUNT PAYABLE  {fmt_money(r['total'], r['currency'], ns, rng, True)}",
    ]
    if r["po_number"]:
        lines.insert(3, f"ORDER REF {r['po_number']}")
    return "\n".join(lines)


def tpl_chat(r, rng):
    """Informal chat message with spelled-out weekday dates and unusual number style."""
    ns = rng.choice(["space", "eu"]) if r["currency"] == "EUR" else "us"
    po = f" PO is {r['po_number']}." if r["po_number"] else ""
    return (
        f"hey {r['contact']} - can you approve this one? {r['vendor']} sent {r['invoice_number']} "
        f"on {fmt_date(r['invoice_date'], 'weekday')}, they want "
        f"{fmt_money(r['total'], r['currency'], ns, rng, True)} all in by "
        f"{fmt_date(r['due_date'], 'weekday')}.{po} it's for {r['customer']}'s account"
    )


OOD_TEMPLATES = [tpl_statement, tpl_chat]


# ------------------------------------------------------------------ dataset assembly

def build_split(name, n, templates, seed, id_offset=0):
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n:
        rec = make_record(rng)
        tpl = templates[len(rows) % len(templates)]
        text = tpl(rec, rng)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"id": f"{name}-{id_offset + len(rows):04d}", "template": tpl.__name__,
                     "text": text, "target": target_of(rec)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    splits = {
        "train": build_split("train", a.train, TRAIN_TEMPLATES, a.seed),
        "val": build_split("val", a.val, TRAIN_TEMPLATES, a.seed + 1),
        "test_id": build_split("test_id", a.test, TRAIN_TEMPLATES, a.seed + 2),
        "test_ood": build_split("test_ood", a.test, OOD_TEMPLATES, a.seed + 3),
    }
    # Guard against leakage: no invoice text may appear in more than one split.
    texts = [r["text"] for rows in splits.values() for r in rows]
    assert len(texts) == len(set(texts)), "duplicate text across splits"
    for name, rows in splits.items():
        write_jsonl(f"{a.out}/{name}.jsonl", rows)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
