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
        # 3. WORKLOAD FUNNEL METRICS (NEW)
        # =========================================================
        
        # A. Transactions handled by Regex (Base for Funnel)
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'regex';
        """)
        regex_handled = cur.fetchone()["c"]
        
        # B. Transactions passed to MiniLM/BERT (Those NOT definitively handled by Regex/Heuristics)
        # Assuming MiniLM processes everything not successfully classified by the previous stage
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'bert';
        """)
        minilm_handled = cur.fetchone()["c"]
        
        # C. Transactions passed to LLM (Those NOT definitively handled by Regex/Heuristics/MiniLM)
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE classification_source = 'llm';
        """)
        llm_handled = cur.fetchone()["c"]

        # =========================================================
        # 4. TEMPORAL FAILURE ANALYSIS (NEW)
        # =========================================================
        
        # Transactions that are PENDING/UNCLEAR over time
        cur.execute("""
            SELECT 
                DATE_TRUNC('month', created_at) AS month_start,
                COUNT(*) AS pending_count
            FROM transactions
            WHERE category = 'PENDING' OR category IS NULL OR category = 'UNCLEAR'
            GROUP BY month_start
            ORDER BY month_start;
        """)
        pending_by_month: List[Dict[str, Any]] = cur.fetchall()


        # =========================================================
        # 5. INDIVIDUAL STAGE PERFORMANCE (Original Queries)
        # =========================================================
        
        # Regex strong
        cur.execute("""
            SELECT COUNT(*) AS c FROM transactions
            WHERE classification_source = 'regex' AND confidence >= 0.8;
        """)
        regex_strong = cur.fetchone()["c"]

        # Regex failed
        cur.execute("""
            SELECT COUNT(*) AS c FROM transactions
            WHERE classification_source = 'regex' AND category = 'PENDING';
        """)
        regex_failed = cur.fetchone()["c"]

        # Heuristics total, strong, failed (assuming 'heuristic' is a valid source)
        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE classification_source = 'heuristic';")
        heur_total = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE classification_source = 'heuristic' AND confidence >= 0.6;")
        heur_strong = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE classification_source = 'heuristic' AND category = 'PENDING';")
        heur_failed = cur.fetchone()["c"]

        # MiniLM strong
        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE classification_source = 'bert' AND confidence >= 0.6;")
        bert_strong = cur.fetchone()["c"]

        # LLM strong and low confidence
        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE classification_source = 'llm' AND confidence >= 0.9;")
        llm_strong = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE classification_source = 'llm' AND confidence < 0.8;")
        llm_low_conf = cur.fetchone()["c"]

        # FINAL PENDING
        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE category = 'PENDING' OR category IS NULL OR category = 'UNCLEAR';")
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
    print("\n[WORKLOAD FUNNEL]:")
    print(f"  Handled by Regex/Heuristics: {regex_handled + heur_total}")
    print(f"  Handled by MiniLM (BERT): {minilm_handled}")
    print(f"  Handled by LLM: {llm_handled}")
    print(f"  Final Unclassified/Pending: {pending_all}")

    # -----------------------------
    print("\n[TEMPORAL FAILURE ANALYSIS]:")
    for row in pending_by_month:
        print(f"  {row['month_start'].strftime('%Y-%m')}: {row['pending_count']} pending")

    # -----------------------------
    print("\n[QUALITY METRICS]:")
    print(f"  Strong regex (conf ≥ 0.8): {regex_strong}")
    print(f"  Strong MiniLM (conf ≥ 0.6): {bert_strong}")
    print(f"  Strong LLM (conf ≥ 0.9): {llm_strong}")
    print(f"  Total PENDING after ALL stages: {pending_all}")
    print("===============================================\n")


if __name__ == "__main__":
    main()
