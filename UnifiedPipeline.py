"""
Unified SpendSight Pipeline
---------------------------
Full ingestion + classification pipeline combining:

 Day 2: PDF parsing → normalized canonical structure
 Day 3: Regex classification
 Day 4: MiniLM semantic classifier

Future (Day 5): LLM fallback can be plugged in where marked.

Execution Flow:
 1. Parse the PDF → raw transactions
 2. Normalize → canonical transaction schema
 3. Insert into DB: documents, statements, transactions
 4. Regex classification (fast deterministic)
 5. MiniLM classification (semantic fallback)
 6. Return classification metrics

This is the main end-to-end processing pipeline.

Author: SpendSight (A-team)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor

# --- Load environment variables ---
load_dotenv()
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Imports from Day 3 Core Pipeline ---
from PipeLine import (
    get_db_connection,
    parse_statement,
    normalize_txn,
    create_document_and_statement,
    insert_transactions,
    update_document_status,
    insert_classification_log
)

# --- Regex classifier ---
from regex_engine.regex_classifier import classify_with_regex

# --- MiniLM classifier ---
from nlp.miniLM_classifier import MiniLMClassifier



# ============================================================
#  Load MiniLM model once (global)
# ============================================================

bert_clf = MiniLMClassifier()
print("[Pipeline] MiniLM classifier initialized.")



# ============================================================
#  Fetch transactions requiring MiniLM classification
# ============================================================

def fetch_transactions_for_minilm(conn, statement_id):
    """
    Returns transactions that require MiniLM classification:
     - category is NULL
     - category = 'PENDING'
     - regex confidence < 0.75
    """
    q = """
        SELECT txn_id, description_clean, description_raw, category, confidence
        FROM transactions
        WHERE statement_id = %s
          AND (
               category IS NULL
            OR category = 'PENDING'
            OR (classification_source = 'regex' AND confidence < 0.75)
          );
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(q, (statement_id,))
        return cur.fetchall()

# ---------------------------------------------------------------------
# CHECK IF STATEMENT ALREADY EXISTS (IDEMPOTENCY FIX)
# ---------------------------------------------------------------------

def statement_exists(conn, user_id, original_filename):
    """
    Returns (True, statement_id) if this PDF has already been processed.
    Checks by matching original filename + user_id.
    """
    query = """
    SELECT s.statement_id
    FROM statements s
    JOIN documents d ON d.doc_id = s.doc_id
    WHERE d.user_id = %s
      AND d.original_filename = %s
    ORDER BY d.upload_time DESC
    LIMIT 1;
    """

    with conn.cursor() as cur:
        cur.execute(query, (user_id, original_filename))
        row = cur.fetchone()
        if row:
            return True, row[0]
        return False, None

# ============================================================
#  Apply MiniLM classification to a single transaction
# ============================================================

def apply_minilm_to_txn(conn, txn):
    """
    Applies MiniLM classification.
    Returns label so pipeline knows whether txn is still pending.
    """
    desc = txn["description_clean"] or txn["description_raw"] or ""
    label, confidence, meta = bert_clf.classify(desc)

    # Split into category.subcategory
    if label == "PENDING":
        category = "PENDING"
        subcategory = None
    else:
        parts = label.split(".")
        category = parts[0]
        subcategory = parts[1] if len(parts) > 1 else None

    # Update DB row
    q = """
        UPDATE transactions
        SET category=%s, subcategory=%s, confidence=%s, classification_source='bert'
        WHERE txn_id=%s
    """
    with conn.cursor() as cur:
        cur.execute(q, (category, subcategory, confidence, txn["txn_id"]))

    # Log classification
    prediction = label
    insert_classification_log(conn, txn["txn_id"], "bert", prediction, confidence, meta)

    conn.commit()
    return label



# ============================================================
#  Main file-level ingestion pipeline
# ============================================================

def process_file(conn, filepath, user_id):
    print(f"\n========== Processing File: {filepath} ==========")
    original_filename = os.path.basename(filepath)

    # ------------------------------------------------------------------
    # IDEMPOTENCY CHECK: If this PDF already processed → skip parsing
    # ------------------------------------------------------------------
    exists, existing_statement_id = statement_exists(conn, user_id, original_filename)

    if exists:
        print(f"[Pipeline] Skipping {original_filename}: already processed (statement_id={existing_statement_id}).")

        # Still run MiniLM classification on remaining unclassified txns
        pending = fetch_transactions_for_minilm(conn, existing_statement_id)

        print(f"[Pipeline] MiniLM needs to process {len(pending)} transactions...")

        mini_attempted = len(pending)
        mini_classified = 0
        mini_pending = 0

        for txn in pending:
            label = apply_minilm_to_txn(conn, txn)
            if label == "PENDING":
                mini_pending += 1
            else:
                mini_classified += 1

        return 0, {
            "mini_attempted": mini_attempted,
            "mini_classified": mini_classified,
            "mini_pending": mini_pending,
        }

    # ----------------------------
    # Step 1: Parse PDF
    # ----------------------------
    bank_name, raw_txns = parse_statement(filepath)
    print(f"[Pipeline] {filepath}: bank={bank_name}, raw={len(raw_txns)}")

    if not bank_name or not raw_txns:
        print("[WARN] No transactions parsed or unknown bank. Skipping.")
        return 0, {
            "mini_attempted": 0,
            "mini_classified": 0,
            "mini_pending": 0
        }

    # ----------------------------
    # Step 2: Insert doc + statement
    # ----------------------------
    doc_id, statement_id = create_document_and_statement(
        conn, user_id, bank_name, filepath, os.path.basename(filepath)
    )

    # ----------------------------
    # Step 3: Normalize transactions
    # ----------------------------
    normalized = []
    for tx in raw_txns:
        n = normalize_txn(tx, statement_id, user_id)
        if n:
            normalized.append(n)
    print(f"[Pipeline] {filepath}: normalized={len(normalized)}")

    # ----------------------------
    # Step 4: Insert transactions
    # ----------------------------
    txn_ids = insert_transactions(conn, normalized)
    print(f"[Pipeline] Inserted {len(txn_ids)} transactions.")

    # ----------------------------
    # Step 5: Regex classifier
    # ----------------------------

    regex_attempted = len(txn_ids)
    regex_classified = 0
    regex_failed = 0

    for i, txn_id in enumerate(txn_ids):
        tx = normalized[i]
        desc = tx["description_raw"]

        category, subcat, vendor, conf, meta = classify_with_regex(desc)

        # If regex matched with confidence > 0
        if category != "PENDING":
            regex_classified += 1
        else:
            regex_failed += 1

        # Update row
        q = """
        UPDATE transactions
        SET vendor=%s, category=%s, subcategory=%s,
            confidence=%s, classification_source='regex'
        WHERE txn_id=%s
        """
        with conn.cursor() as cur:
            cur.execute(q, (vendor, category, subcat, conf, txn_id))
        conn.commit()

        prediction = f"{category}.{subcat}" if subcat else category
        insert_classification_log(conn, txn_id, "regex", prediction, conf, meta)

    print(
        f"[Pipeline][Regex] attempted={regex_attempted}, "
        f"classified={regex_classified}, failed={regex_failed}"
    )

    # ----------------------------
    # Step 6: MiniLM classification
    # ----------------------------
    pending = fetch_transactions_for_minilm(conn, statement_id)
    print(f"[Pipeline] MiniLM evaluating {len(pending)} transactions...")

    mini_attempted = len(pending)
    mini_classified = 0
    mini_pending = 0

    for txn in pending:
        label = apply_minilm_to_txn(conn, txn)
        if label == "PENDING":
            mini_pending += 1
        else:
            mini_classified += 1

    print(
        f"[Pipeline][MiniLM] attempted={mini_attempted}, "
        f"classified={mini_classified}, remaining={mini_pending}"
    )

    # ----------------------------
    # Step 7: Update final status
    # ----------------------------
    update_document_status(conn, doc_id, "parsed")

    return len(txn_ids), {
        "mini_attempted": mini_attempted,
        "mini_classified": mini_classified,
        "mini_pending": mini_pending,
    }



# ============================================================
#  MAIN (process all PDFs in data/input)
# ============================================================

def main():
    input_dir = Path("./data/input")
    pdf_files = sorted([str(p) for p in input_dir.glob("*.pdf")])

    if not pdf_files:
        print("[WARN] No PDFs found in data/input.")
        return

    conn = get_db_connection()

    total_txns = 0
    total_mini_attempted = 0
    total_mini_classified = 0
    total_mini_pending = 0

    for pdf in pdf_files:
        processed, metrics = process_file(conn, pdf, DEFAULT_USER_ID)
        total_txns += processed
        total_mini_attempted += metrics["mini_attempted"]
        total_mini_classified += metrics["mini_classified"]
        total_mini_pending += metrics["mini_pending"]

    conn.close()

    print("\n========= PIPELINE SUMMARY =========")
    print(f"Total transactions inserted      : {total_txns}")
    print(f"MiniLM attempted classifications : {total_mini_attempted}")
    print(f"MiniLM classified (confident)    : {total_mini_classified}")
    print(f"Remaining for LLM fallback       : {total_mini_pending}")
    print("====================================\n")



if __name__ == "__main__":
    main()
