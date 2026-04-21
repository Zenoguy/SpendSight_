import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    """Establishes database connection."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    # Using DATABASE_URL directly assumes it contains all necessary connection parameters
    return psycopg2.connect(DATABASE_URL)


def main():
    conn = get_conn()
    with conn.cursor(cursor_factory=DictCursor) as cur:

        # =========================================================
        # 1. TOTAL TRANSACTIONS
        # =========================================================
        cur.execute("SELECT COUNT(*) AS c FROM transactions;")
        total = cur.fetchone()["c"]

        # =========================================================
        # 2. CLASSIFICATION SOURCE BREAKDOWN (Routing)
        # =========================================================
        cur.execute("""
            SELECT COALESCE(classification_source::text, 'NULL') AS source, COUNT(*) AS c
            FROM transactions
            GROUP BY source
            ORDER BY source;
        """)
        by_source: List[Dict[str, Any]] = cur.fetchall()

        # =========================================================
        # 3. WORKLOAD FUNNEL METRICS
        #    (How many items each stage ultimately owns)
        # =========================================================
        
        # A. Final transactions owned by Regex
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'regex';
        """)
        regex_handled = cur.fetchone()["c"]
        
        # B. Final transactions owned by MiniLM/BERT
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'bert';
        """)
        minilm_handled = cur.fetchone()["c"]
        
        # C. Final transactions owned by LLM
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'llm';
        """)
        llm_handled = cur.fetchone()["c"]

        # D. Final transactions owned by Heuristics
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'heuristic';
        """)
        heur_total = cur.fetchone()["c"]

        # =========================================================
        # 4. TEMPORAL FAILURE / PENDING ANALYSIS
        # =========================================================
        
        cur.execute("""
            SELECT 
                DATE_TRUNC('month', created_at) AS month_start,
                COUNT(*) AS pending_count
            FROM transactions
            WHERE category = 'PENDING' 
               OR category IS NULL 
               OR category = 'UNCLEAR'
            GROUP BY month_start
            ORDER BY month_start;
        """)
        pending_by_month: List[Dict[str, Any]] = cur.fetchall()

        # =========================================================
        # 5. INDIVIDUAL STAGE PERFORMANCE
        # =========================================================
        
        # ---------------- Regex ----------------
        cur.execute("""
            SELECT COUNT(*) AS c 
            FROM transactions
            WHERE classification_source = 'regex' 
              AND confidence >= 0.8;
        """)
        regex_strong = cur.fetchone()["c"]

        # NOTE: after full pipeline, usually regex-owned rows are NOT PENDING,
        # so this will typically be 0 (kept for completeness).
        cur.execute("""
            SELECT COUNT(*) AS c 
            FROM transactions
            WHERE classification_source = 'regex'
              AND category = 'PENDING';
        """)
        regex_failed = cur.fetchone()["c"]

        # ---------------- Heuristics ----------------
        # total already fetched as heur_total above

        cur.execute("""
            SELECT COUNT(*) AS c 
            FROM transactions
            WHERE classification_source = 'heuristic'
              AND confidence >= 0.65;
        """)
        heur_strong = cur.fetchone()["c"]

        cur.execute("""
            SELECT COUNT(*) AS c 
            FROM transactions
            WHERE classification_source = 'heuristic'
              AND category = 'PENDING';
        """)
        heur_failed = cur.fetchone()["c"]

        # ---------------- MiniLM / BERT ----------------
        cur.execute("""
            SELECT COUNT(*) AS c 
            FROM transactions
            WHERE classification_source = 'bert'
              AND confidence >= 0.7;
        """)
        bert_strong = cur.fetchone()["c"]

        # ---------------- LLM ----------------
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

        # ---------------- FINAL PENDING ----------------
        cur.execute("""
            SELECT COUNT(*) AS c 
            FROM transactions
            WHERE category = 'PENDING'
               OR category IS NULL
               OR category = 'UNCLEAR';
        """)
        pending_all = cur.fetchone()["c"]

    conn.close()

    # =========================================================
    # PRINT METRICS
    # =========================================================
    print("\n=========== Classification Breakdown ===========")
    print(f"Total transactions: {total}")

    print("\n[ROUTING SUMMARY]:")
    for row in by_source:
        print(f"  {row['source']:>10}: {row['c']}")

    # -----------------------------
    print("\n[WORKLOAD FUNNEL] (final owner of each txn):")
    print(f"  Handled by Regex          : {regex_handled}")
    print(f"  Handled by Heuristics     : {heur_total}")
    print(f"  Handled by MiniLM (BERT)  : {minilm_handled}")
    print(f"  Handled by LLM            : {llm_handled}")
    print(f"  Final Unclassified/Pending: {pending_all}")

    # -----------------------------
    print("\n[TEMPORAL PENDING ANALYSIS]:")
    if not pending_by_month:
        print("  No PENDING/UNCLEAR transactions over time.")
    else:
        for row in pending_by_month:
            month_label = row['month_start'].strftime('%Y-%m')
            print(f"  {month_label}: {row['pending_count']} pending")

    # -----------------------------
    print("\n[QUALITY METRICS]:")
    print(f"  Strong regex (conf ≥ 0.8)         : {regex_strong}")
    print(f"  Regex still PENDING               : {regex_failed}")

    print(f"\n  Heuristics total                  : {heur_total}")
    print(f"  Strong heuristics (conf ≥ 0.65)    : {heur_strong}")
    print(f"  Heuristics still PENDING          : {heur_failed}")

    print(f"\n  Strong MiniLM (conf ≥ 0.7)        : {bert_strong}")

    print(f"\n  Strong LLM (conf ≥ 0.9)           : {llm_strong}")
    print(f"  LLM low confidence (< 0.8)        : {llm_low_conf}")

    print(f"\n  Total PENDING after ALL stages    : {pending_all}")
    print("===============================================\n")


if __name__ == "__main__":
    main()




