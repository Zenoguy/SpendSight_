import re
from .utils import clean_amount_if_needed

IDBI_TXN_REGEX = re.compile(
    r"""^\s*(\d+)\s+
    (\d{2}/\d{2}/\d{4})\s+
    (\d{2}:\d{2}:\d{2}\s+(?:AM|PM))\s+
    (\d{2}/\d{2}/\d{4})\s+
    (.*?)\s+
    (Cr\.|Dr\.)\s+
    INR\s+([\d,]+\.\d{2})\s+
    ([\d,]+\.\d{2})\s*$""",
    re.VERBOSE
)

def parse_idbi(pdf, filepath):
    print(f"[INFO] Parsing IDBI format for {filepath}...")
    txns = []

    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue

        for line in text.splitlines():
            m = IDBI_TXN_REGEX.match(line)
            if m:
                (
                    serial,
                    date,
                    time,
                    value_date,
                    desc,
                    crdr,
                    amount,
                    balance
                ) = m.groups()

                debit, credit = 0.0, 0.0
                amt = clean_amount_if_needed(amount)

                if crdr == "Dr.":
                    debit = amt
                else:
                    credit = amt

                txns.append({
                    "bank": "IDBI",
                    "date": date,
                    "time": time,
                    "value_date": value_date,
                    "description": desc.strip(),
                    "debit": debit,
                    "credit": credit,
                    "balance": clean_amount_if_needed(balance),
                    "category": None
                })

    print(f"[SUCCESS] Parsed {len(txns)} transactions from {filepath}.")
    return txns
