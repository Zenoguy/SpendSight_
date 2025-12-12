# parsers/idbi.py
import re
import sys
from .utils import clean_amount_if_needed

def parse_idbi(pdf, filepath):
    """
    Permissive IDBI Bank parser (fixed group names).
    """
    print(f"[INFO] Parsing 'IDBI Bank' format for {filepath}...")
    txns = []
    last = None

    IDX = r'(?:\d+\s+)?'               # optional leading index
    DATE_NAMED = r'(?P<date>\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})'
    DATE = r'(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})'   # non-capturing for repeats
    TIME = r'(?:\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM|am|pm)?)?'
    DC = r'(?P<dc>Dr\.?|Cr\.?|DR\.?|CR\.?)'
    AMT = r'(?P<amount>[\(]?[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?[\)]?)'
    BAL = r'(?P<balance>[\d,]+\.\d{2}|\d+(?:,\d{3})*(?:\.\d+)?)'

    p_strict = re.compile(
        rf'^{IDX}{DATE_NAMED}{TIME}\s+{DATE}{TIME}\s+(?P<descr>.+?)\s+{DC}\s*INR\s*{AMT}\s+{BAL}\s*$',
        re.IGNORECASE
    )
    p_alt = re.compile(
        rf'^{IDX}{DATE_NAMED}{TIME}\s+{DATE}{TIME}\s+(?P<descr>.+?)\s+{DC}\s*\.?\s*{AMT}\s+{BAL}\s*$',
        re.IGNORECASE
    )
    p_relaxed = re.compile(
        rf'^{IDX}{DATE_NAMED}{TIME}\s+(?P<descr>.+?)\s+{DC}\s*INR\s*{AMT}\s*(?:\s+{BAL})?\s*$',
        re.IGNORECASE
    )
    p_generic = re.compile(
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

            m = p_strict.match(raw) or p_alt.match(raw) or p_relaxed.match(raw) or p_generic.match(raw)

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
                                credit = amt_val

                    tx = {
                        "bank": "IDBI Bank",
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
                    print(f"[WARN] Skipping malformed IDBI line: {raw} ({e})", file=sys.stderr)
            else:
                if last and raw and not raw.lower().startswith(("idbi bank", "page", "our toll-free", "account no", "primary account")):
                    last["description"] += " " + raw

    print(f"[SUCCESS] Parsed {len(txns)} transactions from {filepath}.")
    return txns
