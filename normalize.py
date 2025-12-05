from datetime import datetime

# ---------------------------------------------------
# Amount Cleaner (required by parsers/utils.py)
# ---------------------------------------------------
def clean_amount(value):
    """
    Convert amount strings like '1,234.56', '(123.45)', '- 123' into float.
    """
    if value is None:
        return 0.0

    val = str(value).strip()

    # Handle negative amounts in parentheses
    if val.startswith("(") and val.endswith(")"):
        val = "-" + val[1:-1]

    # Remove commas & spaces
    val = val.replace(",", "").replace(" ", "")

    try:
        return float(val)
    except:
        return 0.0


# ---------------------------------------------------
# Date Parser
# ---------------------------------------------------
def parse_date(raw_date):
    raw_date = raw_date.strip()

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(raw_date, fmt).date()
        except ValueError:
            pass

    raise ValueError(f"Unrecognized date format: {raw_date!r}")


# ---------------------------------------------------
# Normalizer
# ---------------------------------------------------
def normalize_txn(tx, statement_id, user_id):
    """
    Skip junk rows and normalize real transactions.
    """

    bad_values = (
        "accountholder", "account holder",
        "accountnumber", "account number",
        "statement", "page",
        "opening balance", "closing balance",
        "debit", "credit"
    )

    raw_date = str(tx.get("date", "")).strip().lower()

    # Skip junk rows BEFORE date parsing
    if not raw_date or any(raw_date.startswith(b) for b in bad_values):
        return None

    # Safe date parsing
    try:
        txn_date = parse_date(tx["date"])
    except:
        return None

    description = tx.get("description", "").strip()
    amount = clean_amount(tx.get("amount"))

    return {
        "user_id": user_id,
        "statement_id": statement_id,
        "date": txn_date,
        "description": description,
        "amount": amount,
    }
