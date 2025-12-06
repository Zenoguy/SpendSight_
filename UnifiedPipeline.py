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

from llm.llm_classifier import llm_clf


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



# ---------------------------------------------------------------------
# Fetch transactions needing LLM classification (Fallback)
# ---------------------------------------------------------------------

def fetch_transactions_for_llm(conn, statement_id):
    """
    Fetches transactions that are still PENDING or where the BERT classification 
    confidence is below the LLM fallback threshold (Day 5 requirement).
    """
    # Threshold based on project requirements (e.g., MiniLM confidence < 0.85)
    LLM_FALLBACK_THRESHOLD = 0.50
    
    query = """
    SELECT txn_id, description_clean, description_raw, category, confidence
    FROM transactions
    WHERE statement_id = %s
      AND (
          category = 'PENDING' 
          OR (classification_source = 'bert' AND confidence < %s) -- BERT low confidence
      );
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (statement_id, LLM_FALLBACK_THRESHOLD))
        return cur.fetchall()   

# ---------------------------------------------------------------------
# Apply LLM classification to individual transaction
# ---------------------------------------------------------------------

# Updated apply_llm_to_txn function

def apply_llm_to_txn(conn, txn):
    """
    Sends the transaction to the LLM Fallback, updates the DB, and logs the result.
    """
    desc = txn["description_clean"] or txn["description_raw"] or ""
    
    # 1. Classify using LLM (uses the imported llm_clf object)
    category, subcategory, confidence, meta = llm_clf.classify(desc)

    # 2. Update DB
    q = """
    UPDATE transactions
    SET category=%s, subcategory=%s, confidence=%s, classification_source='llm'
    WHERE txn_id=%s
    """
    with conn.cursor() as cur:
        cur.execute(q, (category, subcategory, confidence, txn["txn_id"]))

    # 3. Log result
    prediction = f"{category}.{subcategory}" if subcategory else category
    insert_classification_log(conn, txn["txn_id"], "llm", prediction, confidence, meta)

    conn.commit()
    # BUG FIX: Must return the classification category for tracking
    return category

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

        # 1) MiniLM on remaining / low-confidence
        pending_bert = fetch_transactions_for_minilm(conn, existing_statement_id)
        print(f"[Pipeline] MiniLM needs to process {len(pending_bert)} transactions...")

        mini_attempted = len(pending_bert)
        mini_classified = 0
        mini_pending = 0

        for txn in pending_bert:
            label = apply_minilm_to_txn(conn, txn)
            if label == "PENDING":
                mini_pending += 1
            else:
                mini_classified += 1

        # 2) LLM on remaining after MiniLM
        llm_pending = fetch_transactions_for_llm(conn, existing_statement_id)
        print(f"[Pipeline] LLM needs to process {len(llm_pending)} transactions...")

        llm_attempted = len(llm_pending)
        llm_classified = 0

        for txn in llm_pending:
            cat = apply_llm_to_txn(conn, txn)
            if cat and cat != "PENDING":
                llm_classified += 1

        print(
            f"[Pipeline][MiniLM] attempted={mini_attempted}, "
            f"classified={mini_classified}, remaining_for_llm={mini_pending}"
        )
        print(
            f"[Pipeline][LLM] attempted={llm_attempted}, "
            f"classified={llm_classified}"
        )

        # No new inserts
        return 0, {
            "mini_attempted": mini_attempted,
            "mini_classified": mini_classified,
            "mini_pending": mini_pending,
            "llm_attempted": llm_attempted,
            "llm_classified": llm_classified,
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
            "mini_pending": 0,
            "llm_attempted": 0,
            "llm_classified": 0,
        }

    # ----------------------------
    # Step 2: Insert doc + statement
    # ----------------------------
    doc_id, statement_id = create_document_and_statement(
        conn, user_id, bank_name, filepath, original_filename
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
    # Step 5: Regex classifier (ALL new txns)
    # ----------------------------
    regex_attempted = len(txn_ids)
    regex_classified = 0
    regex_failed = 0

    for i, txn_id in enumerate(txn_ids):
        tx = normalized[i]
        desc = tx["description_raw"]

        category, subcat, vendor, conf, meta = classify_with_regex(desc)

        if category != "PENDING":
            regex_classified += 1
        else:
            regex_failed += 1

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
    # Step 6: MiniLM classification (ONLY pending/low-conf)
    # ----------------------------
    pending_bert = fetch_transactions_for_minilm(conn, statement_id)
    print(f"[Pipeline] MiniLM evaluating {len(pending_bert)} transactions...")

    mini_attempted = len(pending_bert)
    mini_classified = 0
    mini_pending = 0

    for txn in pending_bert:
        label = apply_minilm_to_txn(conn, txn)
        if label == "PENDING":
            mini_pending += 1
        else:
            mini_classified += 1

    print(
        f"[Pipeline][MiniLM] attempted={mini_attempted}, "
        f"classified={mini_classified}, remaining_for_llm={mini_pending}"
    )

    # Updated LLM Fallback (Step 7) in process_file

    # ----------------------------
    # Step 7: LLM Fallback (ONLY after MiniLM)
    # ----------------------------
    llm_pending = fetch_transactions_for_llm(conn, statement_id)
    print(f"[Pipeline] LLM needs to process {len(llm_pending)} transactions...")

    llm_attempted = len(llm_pending)
    llm_classified = 0

    for txn in llm_pending:
        # Now 'cat' correctly receives the category string from apply_llm_to_txn
        cat = apply_llm_to_txn(conn, txn) 
        
        # Count as classified if it's neither PENDING (default) nor UNCLEAR (LLM failure mode)
        # For simplicity, we count anything that is not PENDING as classified.
        if cat and cat != "PENDING" and cat != "UNCLEAR": # Optional: Exclude UNCLEAR as successful
            llm_classified += 1

    print(
        f"[Pipeline][LLM] attempted={llm_attempted}, "
        f"classified={llm_classified}"
    )

    # ----------------------------
    # Step 8: Update final status
    # ----------------------------
    update_document_status(conn, doc_id, "parsed")

    return len(txn_ids), {
        "mini_attempted": mini_attempted,
        "mini_classified": mini_classified,
        "mini_pending": mini_pending,
        "llm_attempted": llm_attempted,
        "llm_classified": llm_classified,
    }

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
    total_llm_attempted = 0
    total_llm_classified = 0

    for pdf in pdf_files:
        processed, metrics = process_file(conn, pdf, DEFAULT_USER_ID)
        total_txns += processed
        total_mini_attempted += metrics["mini_attempted"]
        total_mini_classified += metrics["mini_classified"]
        total_mini_pending += metrics["mini_pending"]
        total_llm_attempted += metrics.get("llm_attempted", 0)
        total_llm_classified += metrics.get("llm_classified", 0)

    conn.close()

    print("\n========= PIPELINE SUMMARY =========")
    print(f"Total transactions inserted          : {total_txns}")
    print(f"MiniLM attempted classifications     : {total_mini_attempted}")
    print(f"MiniLM classified (confident)        : {total_mini_classified}")
    print(f"Remaining for LLM (after MiniLM)     : {total_mini_pending}")
    print(f"LLM attempted classifications        : {total_llm_attempted}")
    print(f"LLM classified                       : {total_llm_classified}")
    print("====================================\n")

