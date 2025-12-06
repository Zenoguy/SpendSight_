"""
Unified SpendSight Pipeline (Day 3 + Day 4 Combined)
----------------------------------------------------
Steps:
 1. Parse PDF → raw txns
 2. Normalize → canonical fields
 3. Insert into DB
 4. Regex classification → update rows + log
 5. MiniLM classification → update remaining rows + log
"""

import os
import sys
import json
import psycopg2
from psycopg2.extras import DictCursor
from pathlib import Path
from dotenv import load_dotenv
import json
from dotenv import load_dotenv
load_dotenv()

import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("[LLM WARN] GEMINI_API_KEY missing while loading UnifiedPipeline.")
else:
    print("[LLM OK] GEMINI_API_KEY loaded successfully.")



# Import  regex engine
from regex_engine.regex_classifier import classify_with_regex

# Import  MiniLM engine
from nlp.miniLM_classifier import MiniLMClassifier

#Import llm engine
from llm.llm_classifier import llm_clf

# Import  & DB utilities
from PipeLine import (
    get_db_connection,
    parse_statement,
    normalize_txn,
    create_document_and_statement,
    insert_transactions,
    update_document_status,
    insert_classification_log
)

load_dotenv()

DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")
DATABASE_URL = os.getenv("DATABASE_URL")

# ---------------------------------------------------------------------
# MiniLM classifier (loaded once, reused for entire batch)
# ---------------------------------------------------------------------

bert_clf = MiniLMClassifier()
print("[Pipeline] MiniLM classifier loaded.")


# ---------------------------------------------------------------------
# Fetch transactions needing MiniLM classification
# ---------------------------------------------------------------------

def fetch_transactions_for_minilm(conn, statement_id):
    query = """
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
        cur.execute(query, (statement_id,))
        return cur.fetchall()


# ---------------------------------------------------------------------
# Apply MiniLM classification to individual transaction
# ---------------------------------------------------------------------

def apply_minilm_to_txn(conn, txn):
    desc = txn["description_clean"] or txn["description_raw"] or ""
    label, confidence, meta = bert_clf.classify(desc)

    if label == "PENDING":
        category = "PENDING"
        subcategory = None
    else:
        parts = label.split(".")
        category = parts[0]
        subcategory = parts[1] if len(parts) > 1 else None

    # Update DB
    q = """
    UPDATE transactions
    SET category=%s, subcategory=%s, confidence=%s, classification_source='bert'
    WHERE txn_id=%s
    """
    with conn.cursor() as cur:
        cur.execute(q, (category, subcategory, confidence, txn["txn_id"]))

    prediction = label
    insert_classification_log(conn, txn["txn_id"], "bert", prediction, confidence, meta)

    conn.commit()

# ---------------------------------------------------------------------
# Fetch transactions needing LLM classification (Fallback)
# ---------------------------------------------------------------------

def fetch_transactions_for_llm(conn, statement_id):
    """
    Fetches transactions that are still PENDING or where the BERT classification 
    confidence is below the LLM fallback threshold (Day 5 requirement).
    """
    # Threshold based on project requirements (e.g., MiniLM confidence < 0.85)
    LLM_FALLBACK_THRESHOLD = 0.85
    
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

def apply_llm_to_txn(conn, txn):
    """
    Sends the transaction to the LLM Fallback, updates the DB, and logs the result.
    """
    desc = txn["description_clean"] or txn["description_raw"] or ""
    
    # 1. Classify using LLM (uses the imported llm_clf object)
    # The llm_clf.classify function now returns category, subcategory, confidence, meta
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
    # Assuming insert_classification_log is defined in PipeLine.py or imported
    insert_classification_log(conn, txn["txn_id"], "llm", prediction, confidence, meta)

    conn.commit()


# ---------------------------------------------------------------------
# PROCESS A PDF FILE COMPLETELY
# ---------------------------------------------------------------------

def process_file(conn, filepath, user_id):
    print(f"\n[Pipeline] Processing file: {filepath}")

    # Step 1: Use the parsing system (Day 3)
    bank_name, raw_txns = parse_statement(filepath)
    if not bank_name or not raw_txns:
        print("[WARN] Could not parse PDF.")
        return 0

    # Step 2: Create document + statement rows
    doc_id, statement_id = create_document_and_statement(
        conn, user_id, bank_name, filepath, os.path.basename(filepath)
    )

    # Step 3: Normalize txns
    normalized = []
    for tx in raw_txns:
        nt = normalize_txn(tx, statement_id, user_id)
        if nt:
            normalized.append(nt)

    # Step 4: INSERT into DB
    txn_ids = insert_transactions(conn, normalized)
    print(f"[Pipeline] Inserted {len(txn_ids)} transactions.")

    # Step 5: Apply regex classification (Day 3)
    for i, txn_id in enumerate(txn_ids):
        tx = normalized[i]
        desc = tx["description_raw"]

        category, subcat, vendor, conf, meta = classify_with_regex(desc)

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

    print("[Pipeline] Regex classification complete.")

    # Step 6: Apply MiniLM to unclassified / low-confidence txns (Day 4)
    pending = fetch_transactions_for_minilm(conn, statement_id)
    print(f"[Pipeline] MiniLM needs to process {len(pending)} transactions...")

    for txn in pending:
        apply_minilm_to_txn(conn, txn)

    print("[Pipeline] MiniLM classification complete.")

    # Step 7: Apply LLM Fallback (Day 5)
    llm_pending = fetch_transactions_for_llm(conn, statement_id)
    print(f"[Pipeline] LLM needs to process {len(llm_pending)} transactions...")

    for txn in llm_pending:
        apply_llm_to_txn(conn, txn)

    print("[Pipeline] LLM Fallback classification complete.")
    
    # Step 8: Update document status
    update_document_status(conn, doc_id, "parsed")

    return len(txn_ids)


# ---------------------------------------------------------------------
# MAIN PIPELINE EXECUTION (ALL PDFs IN FOLDER)
# ---------------------------------------------------------------------

def main():
    input_dir = Path("./data/input")
    pdf_files = [str(p) for p in input_dir.glob("*.pdf")]

    conn = get_db_connection()
    total = 0

    for f in pdf_files:
        total += process_file(conn, f, DEFAULT_USER_ID)

    conn.close()
    print(f"\n[Pipeline] Finished. Transactions processed: {total}")

if __name__ == "__main__":
    main()
