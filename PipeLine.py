import os
import sys
import json
from pathlib import Path
from datetime import datetime
import uuid
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv

# Day 3: Regex engine
from regex_engine.regex_classifier import classify_with_regex

# Your existing parsers
from parsers.bob import parse_bob
from parsers.pnb import parse_pnb
from parsers.sbi import parse_sbi
from parsers.federal import parse_federal_bank

load_dotenv()

DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")
DATABASE_URL = os.getenv("DATABASE_URL")

# --------------------------------------------------------
# DB CONNECTION
# --------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# --------------------------------------------------------
# DB HELPERS
# --------------------------------------------------------

def insert_transactions(conn, transactions):
    if not transactions:
        return []

    columns = transactions[0].keys()
    values = [tuple(tx.values()) for tx in transactions]

    query = f"""
    INSERT INTO transactions ({', '.join(columns)})
    VALUES %s
    RETURNING txn_id;
    """

    with conn.cursor() as cur:
        extras.execute_values(cur, query, values)
        inserted_ids = [row[0] for row in cur.fetchall()]
        conn.commit()
        return inserted_ids


def insert_classification_log(conn, txn_id, stage, prediction, confidence, meta):
    q = """
    INSERT INTO classification_log (txn_id, stage, prediction, confidence, meta)
    VALUES (%s, %s, %s, %s, %s::jsonb)
    """
    with conn.cursor() as cur:
        cur.execute(q, (txn_id, stage, prediction, confidence, json.dumps(meta)))
    conn.commit()


def create_document_and_statement(conn, user_id, bank_name, file_path, original_filename):
    doc_q = """
    INSERT INTO documents (user_id, doc_type, file_path, original_filename, status)
    VALUES (%s, 'bank_statement', %s, %s, 'uploaded')
    RETURNING doc_id;
    """

    stmt_q = """
    INSERT INTO statements (doc_id, user_id, bank_name, status)
    VALUES (%s, %s, %s, 'parsed')
    RETURNING statement_id;
    """

    with conn.cursor() as cur:
        cur.execute(doc_q, (user_id, file_path, original_filename))
        doc_id = cur.fetchone()[0]

        cur.execute(stmt_q, (doc_id, user_id, bank_name))
        statement_id = cur.fetchone()[0]

    conn.commit()
    return doc_id, statement_id


def update_document_status(conn, doc_id, status):
    q = "UPDATE documents SET status = %s WHERE doc_id = %s"
    with conn.cursor() as cur:
        cur.execute(q, (status, doc_id))
    conn.commit()

# --------------------------------------------------------
# PDF DETECTION + ROUTING
# --------------------------------------------------------

def parse_statement(filepath):
    import pdfplumber

    with pdfplumber.open(filepath) as pdf:
        first = (pdf.pages[0].extract_text() or "").lower()

        if "bank of baroda" in first or "statement of account"  in first:
            return "BOB", parse_bob(pdf, filepath)
        if "punjab national bank" in first:
            return "PNB", parse_pnb(pdf, filepath)
        if "state bank of india" in first or "sbi" in first:
            return "SBI", parse_sbi(pdf, filepath)
        if "federal bank" in first:
            return "Federal Bank", parse_federal_bank(pdf, filepath)

        return None, []

# --------------------------------------------------------
# NORMALIZATION
# --------------------------------------------------------

def clean_amount(v):
    if not v:
        return 0.0
    s = str(v).replace(",", "").replace(" ", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except:
        return 0.0


def parse_date(raw):
    raw = raw.strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except:
            pass
    return None


def normalize_txn(tx, statement_id, user_id):
    raw_date = str(tx.get("date", "")).strip()
    if not raw_date or len(raw_date) < 4:
        return None

    d = parse_date(raw_date)
    if not d:
        return None

    desc = tx.get("description", "").strip()
    amt = clean_amount(tx.get("amount"))

    return {
        "user_id": user_id,
        "statement_id": statement_id,
        "txn_date": d,
        "description_raw": desc,
        "amount": amt,
        "vendor": None,
        "category": None,
        "subcategory": None,
        "confidence": 0.0,
        "classification_source": None,
    }

# --------------------------------------------------------
# MAIN PIPELINE PER FILE
# --------------------------------------------------------

def process_pdf(conn, filepath, user_id):
    print(f"\n--- Processing: {os.path.basename(filepath)}")

    bank, raw_txns = parse_statement(filepath)
    if not bank or not raw_txns:
        print("[WARN] Could not detect bank or parse PDF.")
        return 0

    doc_id, statement_id = create_document_and_statement(
        conn, user_id, bank, filepath, os.path.basename(filepath)
    )

    normalized = []
    for tx in raw_txns:
        n = normalize_txn(tx, statement_id, user_id)
        if n:
            normalized.append(n)

    # INSERT FIRST to get txn_ids
    txn_ids = insert_transactions(conn, normalized)

    # APPLY REGEX CLASSIFICATION
    for idx, txn_id in enumerate(txn_ids):
        tx = normalized[idx]
        desc = tx["description_raw"]

        category, subcategory, vendor, conf, meta = classify_with_regex(desc)

        # UPDATE transactions
        q = """
        UPDATE transactions
        SET vendor=%s, category=%s, subcategory=%s, confidence=%s, classification_source='regex'
        WHERE txn_id=%s
        """
        with conn.cursor() as cur:
            cur.execute(q, (vendor, category, subcategory, conf, txn_id))
        conn.commit()

        # INSERT INTO LOG
        prediction = f"{category}.{subcategory}" if category else None
        insert_classification_log(conn, txn_id, "regex", prediction, conf, meta)

    update_document_status(conn, doc_id, "parsed")
    return len(txn_ids)

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main():
    if not DEFAULT_USER_ID:
        print("[ERROR] DEFAULT_USER_ID not set")
        sys.exit(1)

    input_dir = Path("./data/input")
    files = [str(p) for p in input_dir.glob("*.pdf")]

    print("SpendSight – Day 3 Pipeline\n")

    conn = get_db_connection()
    total = 0

    for f in files:
        total += process_pdf(conn, f, DEFAULT_USER_ID)

    conn.close()

    print(f"\nTotal processed: {total}")


if __name__ == "__main__":
    main()
