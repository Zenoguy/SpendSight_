# db.py
import os
import json
import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL=os.getenv("DATABASE_URL")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "finparse")
DB_USER = os.getenv("DB_USER", "finuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "finpass")


def get_db_connection():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL, sslmode="require")
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def create_document_and_statement(conn, user_id, bank_name, file_path, original_filename):
    """
    Inserts into documents + statements and returns (doc_id, statement_id).
    For now period_start/end & account_number are left NULL.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (user_id, doc_type, file_path, original_filename, status, meta)
            VALUES (%s, 'bank_statement', %s, %s, 'uploaded', %s)
            RETURNING doc_id
            """,
            (user_id, file_path, original_filename, json.dumps({"bank": bank_name})),
        )
        doc_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO statements (doc_id, user_id, period_start, period_end,
                                    account_number, bank_name, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'parsed')
            RETURNING statement_id
            """,
            (doc_id, user_id, None, None, None, bank_name),
        )
        statement_id = cur.fetchone()[0]

    conn.commit()
    return doc_id, statement_id


def insert_transactions(conn, normalized_txns):
    """
    Bulk insert normalized transactions into the transactions table.
    """
    if not normalized_txns:
        return

    cols = (
        "statement_id",
        "user_id",
        "txn_date",
        "posting_date",
        "description_raw",
        "description_clean",
        "amount",
        "direction",
        "vendor",
        "category",
        "subcategory",
        "confidence",
        "classification_source",
    )

    values = [
        (
            tx["statement_id"],
            tx["user_id"],
            tx["txn_date"],
            tx["posting_date"],
            tx["description_raw"],
            tx["description_clean"],
            tx["amount"],
            tx["direction"],
            tx["vendor"],
            tx["category"],
            tx["subcategory"],
            tx["confidence"],
            tx["classification_source"],
        )
        for tx in normalized_txns
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO transactions ({", ".join(cols)})
            VALUES %s
            """,
            values,
        )
    conn.commit()
