
# debug_amounts.py
import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

conn = psycopg2.connect(DATABASE_URL)

with conn, conn.cursor(cursor_factory=DictCursor) as cur:
    print("=== Basic amount stats ===")
    cur.execute("""
        SELECT
            COUNT(*) AS total,
            MIN(amount) AS min_amount,
            MAX(amount) AS max_amount,
            SUM(amount) AS sum_amount
        FROM transactions;
    """)
    print(cur.fetchone())

    print("\n=== Non-zero amounts sample (if any) ===")
    cur.execute("""
        SELECT txn_id, txn_date, amount, category, vendor
        FROM transactions
        WHERE amount <> 0
        LIMIT 10;
    """)
    rows = cur.fetchall()
    for row in rows:
        print(dict(row))

    print("\n=== Category-wise sum of amounts ===")
    cur.execute("""
        SELECT COALESCE(category, 'NULL') AS category,
               COUNT(*) AS n,
               SUM(amount) AS sum_amount
        FROM transactions
        GROUP BY COALESCE(category, 'NULL')
        ORDER BY sum_amount DESC NULLS LAST;
    """)
    for row in cur:
        print(dict(row))

conn.close()

