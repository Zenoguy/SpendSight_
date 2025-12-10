import re
import sys
from .utils import clean_amount_if_needed

# MAIN TRANSACTION REGEX
TXN_REGEX = re.compile(
    r"^(?P<date>\d{2}-\d{2}-\d{4})\s+"
    r"(?P<mode>[A-Z0-9 /.-]+?)\s+"
    r"(?P<particulars>.+?)\s+"
    r"(?P<deposit>\d[\d,]*\.\d{2}|0)\s+"
    r"(?P<withdrawal>\d[\d,]*\.\d{2}|0)\s+"
    r"(?P<balance>\d[\d,]*\.\d{2})$"
)

# CONTINUATION LINE REGEX
CONT_REGEX = re.compile(r"^(?!\d{2}-\d{2}-\d{4})(?P<cont>.+)$")


def parse_icici(pdf, filepath):
    print(f"[INFO] Parsing ICICI Bank format for {filepath}...")
    txns = []
    current = None

    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue

        for line in text.split("\n"):
            line = line.strip()

            # Match new transaction
            m = TXN_REGEX.match(line)
            if m:
                if current:
                    txns.append(current)

                gd = m.groupdict()
                current = {
                    "bank": "ICICI",
                    "date": gd["date"],
                    "mode": gd["mode"].strip(),
                    "description": gd["particulars"].strip(),
                    "credit": clean_amount_if_needed(gd["deposit"]),
                    "debit": clean_amount_if_needed(gd["withdrawal"]),
                    "balance": clean_amount_if_needed(gd["balance"]),
                    "category": None,
                }
                continue

            # Match continuation of particulars
            c = CONT_REGEX.match(line)
            if c and current:
                current["description"] += " " + c.group("cont").strip()

    if current:
        txns.append(current)

    print(f"[SUCCESS] Parsed {len(txns)} transactions from {filepath}.")
    return txns
