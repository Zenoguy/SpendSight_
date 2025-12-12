# parsers/icici.py
import re
import sys
from .utils import clean_amount_if_needed

def parse_icici(pdf, filepath):
    """
    Permissive ICICI Bank parser (fixed group names).
    """
    print(f"[INFO] Parsing 'ICICI Bank' format for {filepath}...")
    txns = []
    last = None

    IDX = r'(?:\d+\s+)?'
    DATE_NAMED = r'(?P<date>\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})'
    DATE = r'(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})'
    TIME = r'(?:\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM|am|pm)?)?'
    DC = r'(?P<dc>Dr\.?|Cr\.?|DR|CR|Cr\.)'
    AMT = r'(?P<amount>[\(]?[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?[\)]?)'
    BAL = r'(?P<balance>[\d,]+\.\d{2}|\d+(?:,\d{3})*(?:\.\d+)?)'

    p1 = re.compile(
        rf'^{IDX}{DATE_NAMED}{TIME}\s+{DATE}{TIME}\s+(?P<descr>.+?)\s+{DC}\s*\.?\s*(?:INR\s*)?{AMT}\s+{BAL}\s*$',
        re.IGNORECASE
    )
    p2 = re.compile(
        rf'^{IDX}{DATE_NAMED}{TIME}\s+(?P<descr>.+?)\s+{DC}\s*\.?\s*(?:INR\s*)?{AMT}\s*(?:\s+{BAL})?\s*$',
        re.IGNORECASE
    )
    p3 = re.compile(
        rf'^{IDX}{DATE_NAMED}{TIME}\s+(?P<descr>.+?)\s+{AMT}\s*(?:\s+{BAL})?\s*$',
        re.IGNORECASE
    )

    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue
        lines = text.splitlines()
        for line in lines:
            raw = line.strip()
            if not raw:
                continue

            m = p1.match(raw) or p2.match(raw) or p3.match(raw)

            if m:
                try:
                    d = (m.group("date") or "").strip()
                    descr = (m.group("descr") or "").strip()
                    amt_raw = m.groupdict().get("amount") or ""
                    bal_raw = m.groupdict().get("balance") or None
                    dc_tok = m.groupdict().get("dc") or ""

                    amt_val = clean_amount_if_needed(amt_raw)
                    bal_val = clean_amount_if_needed(bal_raw) if bal_raw else None

                    debit = 0.0
                    credit = 0.0

                    if isinstance(amt_val, float) and (str(amt_raw).strip().startswith("(") or str(amt_raw).strip().startswith("-")):
                        debit = abs(amt_val)
                    else:
                        if dc_tok:
                            if dc_tok.strip().upper().startswith("D"):
                                debit = amt_val
                            else:
                                credit = amt_val
                        else:
                            if re.search(r'\bCR\b', descr, flags=re.IGNORECASE):
                                credit = amt_val
                                descr = re.sub(r'\bCR\b', '', descr, flags=re.IGNORECASE).strip()
                            elif re.search(r'\bDR\b', descr, flags=re.IGNORECASE):
                                debit = amt_val
                                descr = re.sub(r'\bDR\b', '', descr, flags=re.IGNORECASE).strip()
                            else:
                                if re.search(r'\bNEFT|IMPS|CREDIT|CR\b', descr, flags=re.IGNORECASE):
                                    credit = amt_val
                                elif re.search(r'UPI|VISA|POS|ATM|PAY|PHONEPE|PAYTM|HOTEL|RESTAURANT|STORE|GROCERY', descr, flags=re.IGNORECASE):
                                    debit = amt_val
                                else:
                                    # conservative default: mark as debit (most merchant lines are debits)
                                    debit = amt_val

                    tx = {
                        "bank": "ICICI Bank",
                        "date": d,
                        "description": descr,
                        "debit": debit,
                        "credit": credit,
                        "balance": bal_val,
                        "category": None,
                    }
                    txns.append(tx)
                    last = tx
                except Exception as e:
                    print(f"[WARN] Skipping malformed ICICI line: {raw} ({e})", file=sys.stderr)
            else:
                if last and raw and not raw.lower().startswith(("page", "statement", "account", "icici", "ifsc")):
                    last["description"] += " " + raw

    print(f"[SUCCESS] Parsed {len(txns)} transactions from {filepath}.")
    return txns
