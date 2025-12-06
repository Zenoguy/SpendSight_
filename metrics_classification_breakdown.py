# metrics_classification_breakdown.py

import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def main():
    conn = get_conn()
    with conn.cursor(cursor_factory=DictCursor) as cur:
        # Total
        cur.execute("SELECT COUNT(*) AS c FROM transactions;")
        total = cur.fetchone()["c"]

        # By classification_source
        cur.execute("""
            SELECT classification_source, COUNT(*) AS c
            FROM transactions
            GROUP BY classification_source
            ORDER BY classification_source;
        """)
        by_source = cur.fetchall()

        # Regex with decent confidence (e.g., >= 0.8)
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'regex'
              AND confidence >= 0.8;
        """)
        regex_strong = cur.fetchone()["c"]

        # Regex that "failed" (still PENDING after regex)
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'regex'
              AND (category = 'PENDING' OR category IS NULL);
        """)
        regex_failed = cur.fetchone()["c"]

        # Bert (MiniLM) strong classifications
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'bert'
              AND confidence >= 0.6;
        """)
        bert_strong = cur.fetchone()["c"]

        # Still pending after all stages (LLM candidates)
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE category = 'PENDING'
               OR category IS NULL;
        """)
        pending_all = cur.fetchone()["c"]

    conn.close()

    print("=========== Classification Breakdown ===========")
    print(f"Total transactions: {total}")
    print("\nBy classification_source:")
    for row in by_source:
        print(f"  {row['classification_source'] or 'NULL':>6}: {row['c']}")

    print("\nRegex performance:")
    print(f"  Strong regex (conf >= 0.8): {regex_strong}")
    print(f"  Regex 'failed' (still PENDING): {regex_failed}")

    print("\nMiniLM performance:")
    print(f"  Strong bert (conf >= 0.6): {bert_strong}")

    print("\nOverall remaining PENDING (LLM candidates):")
    print(f"  Pending: {pending_all}")
    print("===============================================")


if __name__ == "__main__":
    main()
