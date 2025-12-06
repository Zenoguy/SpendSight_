# phase4_minilm_batch.py

import os
import json
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

from miniLM_classifier import MiniLMClassifier

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def fetch_pending_transactions(conn, limit=300):
    """
    Fetch transactions that still need NLP classification.
    - no category OR 'PENDING'
    OR
    - only regex classification with low confidence
    """
    query = """
    SELECT
        txn_id,
        description_clean,
        description_raw,
        category,
        confidence,
        classification_source
    FROM transactions
    WHERE
        (category IS NULL OR category = 'PENDING')
        OR
        (classification_source = 'regex' AND confidence < 0.75)
    ORDER BY txn_date ASC
    LIMIT %s;
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (limit,))
        return cur.fetchall()


def update_transaction_with_bert(conn, txn_id, label, confidence, meta):
    """
    Update transactions table + classification_log with BERT/MiniLM result.
    """
    if label == "PENDING":
        category = "PENDING"
        subcategory = None
    else:
        parts = label.split(".")
        category = parts[0]
        subcategory = parts[1] if len(parts) > 1 else None

    # 1) Update transactions
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE transactions
            SET category = %s,
                subcategory = %s,
                confidence = %s,
                classification_source = 'bert'
            WHERE txn_id = %s
            """,
            (category, subcategory, confidence, txn_id),
        )

    # 2) Insert into classification_log
    prediction = f"{category}.{subcategory}" if subcategory else category

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO classification_log (txn_id, stage, prediction, confidence, meta)
            VALUES (%s, 'bert', %s, %s, %s::jsonb)
            """,
            (txn_id, prediction, confidence, json.dumps(meta)),
        )

    conn.commit()


def main():
    print("[PHASE4] Loading MiniLM classifier...")
    clf = MiniLMClassifier()
    conn = get_db_conn()

    try:
        pending = fetch_pending_transactions(conn, limit=300)
        print(f"[PHASE4] Found {len(pending)} eligible transactions for MiniLM.")

        for row in pending:
            txn_id = row["txn_id"]
            desc = row["description_clean"] or row["description_raw"] or ""

            label, conf, meta = clf.classify(desc)

            update_transaction_with_bert(conn, txn_id, label, conf, meta)

        print("[PHASE4] MiniLM classification complete.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
