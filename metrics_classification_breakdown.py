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
    return psycopg2.connect(DATABASE_URL)


def main():
    conn = get_conn()
    with conn.cursor(cursor_factory=DictCursor) as cur:

        # =========================================================
        # TOTAL TRANSACTIONS
        # =========================================================
        cur.execute("SELECT COUNT(*) AS c FROM transactions;")
        total = cur.fetchone()["c"]

        # =========================================================
        # CLASSIFICATION SOURCE BREAKDOWN
        # =========================================================
        cur.execute("""
            SELECT COALESCE(classification_source::text, 'NULL') AS source, COUNT(*) AS c
            FROM transactions
            GROUP BY source
            ORDER BY source;
        """)
        by_source = cur.fetchall()

        # =========================================================
        # REGEX PERFORMANCE
        # =========================================================
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'regex'
              AND confidence >= 0.8;
        """)
        regex_strong = cur.fetchone()["c"]

        # Regex failed = regex assigned but still PENDING
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'regex'
              AND category = 'PENDING';
        """)
        regex_failed = cur.fetchone()["c"]

        # =========================================================
        # HEURISTICS PERFORMANCE
        # =========================================================
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'heuristic';
        """)
        heur_total = cur.fetchone()["c"]

        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'heuristic'
              AND confidence >= 0.6;
        """)
        heur_strong = cur.fetchone()["c"]

        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'heuristic'
              AND category = 'PENDING';
        """)
        heur_failed = cur.fetchone()["c"]

        # =========================================================
        # MiniLM (BERT) PERFORMANCE
        # =========================================================
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'bert'
              AND confidence >= 0.6;
        """)
        bert_strong = cur.fetchone()["c"]

        # =========================================================
        # LLM PERFORMANCE
        # =========================================================
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'llm'
              AND confidence >= 0.9;
        """)
        llm_strong = cur.fetchone()["c"]

        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'llm'
              AND confidence < 0.8;
        """)
        llm_low_conf = cur.fetchone()["c"]

        # =========================================================
        # FINAL PENDING
        # =========================================================
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE category = 'PENDING'
               OR category IS NULL;
        """)
        pending_all = cur.fetchone()["c"]

    conn.close()

    # =========================================================
    # PRINT METRICS
    # =========================================================
    print("\n=========== Classification Breakdown ===========")
    print(f"Total transactions: {total}")

    print("\nBy classification_source:")
    for row in by_source:
        print(f"  {row['source']:>10}: {row['c']}")

    # -----------------------------
    print("\nRegex performance:")
    print(f"  Strong regex (conf ≥ 0.8): {regex_strong}")
    print(f"  Regex failed (still PENDING): {regex_failed}")

    # -----------------------------
    print("\nHeuristics performance:")
    print(f"  Total heuristics classified: {heur_total}")
    print(f"  Strong heuristics (conf ≥ 0.6): {heur_strong}")
    print(f"  Heuristics failed (still PENDING): {heur_failed}")

    # -----------------------------
    print("\nMiniLM (BERT) performance:")
    print(f"  Strong MiniLM (conf ≥ 0.6): {bert_strong}")

    # -----------------------------
    print("\nLLM performance:")
    print(f"  Strong LLM (conf ≥ 0.9): {llm_strong}")
    print(f"  LLM low confidence (< 0.8): {llm_low_conf}")

    # -----------------------------
    print("\nFINAL: Remaining PENDING after ALL stages:")
    print(f"  Pending: {pending_all}")
    print("===============================================\n")


if __name__ == "__main__":
    main()
