import re
from datetime import datetime

def parse_ocr_generic(text):
    """
    Parses OCR-extracted text that may not belong to any known bank.
    Looks for generic patterns: dates + amounts + description.
    Returns list of transaction dicts.
    """

    # Common date regexes
    date_patterns = [
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}\b"  # fallback
    ]

    amount_patterns = [
        r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?",
        r"₹\s*\d+(?:,\d+)*(?:\.\d+)?"
    ]

    # Combine date + anything + amount
    line_regex = re.compile(
        rf"({'|'.join(date_patterns)})\s+(.*?)\s+({'|'.join(amount_patterns)})",
        re.IGNORECASE
    )

    txns = []

    for match in line_regex.finditer(text):
        date_str, desc, amt_str = match.groups()

        # Normalize date
        try:
            txn_date = datetime.strptime(date_str, "%d/%m/%Y")
        except:
            try:
                txn_date = datetime.strptime(date_str, "%Y-%m-%d")
            except:
                continue

        # Normalize amount
        amt_clean = re.sub(r"[₹,]", "", amt_str)
        amount = float(amt_clean)

        # Direction
        direction = "debit" if "-" in amt_str else "credit"

        txns.append({
            "txn_date": txn_date.date(),
            "posting_date": None,
            "description_raw": desc.strip(),
            "description_clean": desc.strip().lower(),
            "amount": amount if direction == "debit" else -amount,
            "direction": direction,
            "vendor": None,
            "category": "PENDING",
            "subcategory": None,
            "confidence": 0.0,
            "classification_source": "pending"
        })

    return txns
