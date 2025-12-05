# main_pipeline.py
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path

from db import get_db_connection, create_document_and_statement, insert_transactions
from normalize import normalize_txn
from parsers import parse_statement


DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")


def process_pdf(conn, filepath: str, user_id):
    print(f"\nProcessing: {os.path.basename(filepath)}")

    bank_name, raw_txns = parse_statement(filepath)
    if not bank_name or not raw_txns:
        print("[WARN] No transactions parsed or unknown bank.")
        return 0

    # Create document & statement record
    doc_id, statement_id = create_document_and_statement(
        conn,
        user_id=user_id,
        bank_name=bank_name,
        file_path=filepath,
        original_filename=os.path.basename(filepath),
    )

    # Normalize transactions (filter out skipped/None results)
    normalized = []
    for tx in raw_txns:
        normalized_tx = normalize_txn(tx, statement_id=statement_id, user_id=user_id)
        if normalized_tx is not None:
            normalized.append(normalized_tx)

    # Insert into DB
    insert_transactions(conn, normalized)
    print(f"[INFO] Inserted {len(normalized)} transactions into DB.")
    return len(normalized)


def main(pdf_directory="./data/input"):
    if not DEFAULT_USER_ID:
        print(
            "[ERROR] DEFAULT_USER_ID not set. "
            "Set it to an existing users.user_id (UUID) in your database.",
            file=sys.stderr,
        )
        sys.exit(1)

    pdf_dir = Path(pdf_directory)
    if not pdf_dir.is_dir():
        print(f"[ERROR] PDF directory does not exist: {pdf_dir}")
        sys.exit(1)

    pdf_files = sorted([str(p) for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])

    if not pdf_files:
        print(f"[ERROR] No PDF files found in {pdf_dir}")
        sys.exit(1)

    print("\n====================================")
    print("SpendSight – Parsing & DB Ingestion")
    print("====================================\n")
    print(f"Found {len(pdf_files)} PDF file(s):")
    for f in pdf_files:
        print("  -", os.path.basename(f))

    conn = get_db_connection()
    total_inserted = 0

    try:
        for f in pdf_files:
            total_inserted += process_pdf(conn, f, user_id=DEFAULT_USER_ID)
    finally:
        conn.close()

    print("\n------------------------------------")
    print(f"Total transactions inserted: {total_inserted}")
    print("------------------------------------\n")


if __name__ == "__main__":
    main()
