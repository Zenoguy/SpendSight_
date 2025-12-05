import os
import sys
from pathlib import Path
from datetime import datetime
import uuid # For generating mock IDs and handling UUID types
import psycopg2 # Assuming PostgreSQL/psycopg2 for DB connection
from psycopg2 import extras # For dictionary cursor
from dotenv import load_dotenv

# Load environment variables (e.g., DEFAULT_USER_ID, DB credentials)
load_dotenv() 

# --- CONFIGURATION ---
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")
BUCKET_BASE_DIR = Path("./data/bucket") # Local directory simulating S3/GCS bucket

# ====================================================================
# I. DB UTILITIES (Conceptual Implementation)
# These functions handle interaction with the Postgres database.
# ====================================================================

def get_db_connection():
    """Connect using DATABASE_URL if present, otherwise fallback."""
    db_url = os.getenv("DATABASE_URL")

    # If DATABASE_URL exists, use it
    if db_url:
        try:
            conn = psycopg2.connect(db_url, sslmode="require")
            print("[DB] Connected using DATABASE_URL")
            return conn
        except Exception as e:
            print(f"[DB ERROR] Failed to connect using DATABASE_URL: {e}")
            sys.exit(1)

    # Fallback to old method
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "secret"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        print("[DB] Connected using individual DB credentials")
        return conn

    except psycopg2.Error as e:
        print(f"[DB ERROR] Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)


def insert_transactions(conn: psycopg2.extensions.connection, transactions: list[dict]):
    """Inserts a list of normalized transactions into the transactions table."""
    if not transactions:
        return

    columns = (
        "user_id",
        "statement_id",
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
            t["user_id"],
            t["statement_id"],
            t["txn_date"],
            t["posting_date"],
            t["description_raw"],
            t["description_clean"],
            t["amount"],
            t["direction"],
            t["vendor"],
            t["category"],
            t["subcategory"],
            t["confidence"],
            t["classification_source"],
        )
        for t in transactions
    ]

    query = f"""
    INSERT INTO transactions ({', '.join(columns)})
    VALUES %s;
    """

    with conn.cursor() as cur:
        try:
            extras.execute_values(cur, query, values)
            conn.commit()
        except psycopg2.Error as e:
            print(f"[DB ERROR] Could not insert transactions: {e}", file=sys.stderr)
            conn.rollback()
            raise


def create_document_and_statement(conn, user_id, bank_name, file_path, original_filename) -> tuple[uuid.UUID, uuid.UUID]:
    """
    STUB: Creates a row in documents and a corresponding row in statements.
    
    This fulfills the requirement for the File Upload System (Step 3).
    Returns: (doc_id, statement_id)
    """
    
    # === 1. Insert into documents table ===
    doc_query = """
    INSERT INTO documents 
        (user_id, doc_type, file_path, original_filename, status)
    VALUES 
        (%s, %s, %s, %s, 'uploaded')
    RETURNING doc_id;
    """
    
    doc_id = None
    with conn.cursor() as cur:
        # Use 'bank_statement' as the doc_type, assuming PDF parsing focuses on this
        cur.execute(doc_query, (user_id, 'bank_statement', file_path, original_filename))
        doc_id = cur.fetchone()[0]
    
    # === 2. Insert into statements table ===
    statement_query = """
    INSERT INTO statements 
        (doc_id, user_id, bank_name, status)
    VALUES 
        (%s, %s, %s, 'parsed')
    RETURNING statement_id;
    """
    
    statement_id = None
    with conn.cursor() as cur:
        cur.execute(statement_query, (doc_id, user_id, bank_name))
        statement_id = cur.fetchone()[0]
        conn.commit()

    print(f"[DB] Created Document ID: {doc_id} and Statement ID: {statement_id}")
    return doc_id, statement_id


def update_document_status(conn: psycopg2.extensions.connection, doc_id: uuid.UUID, new_status: str):
    """Updates the status of a document (e.g., from 'uploaded' to 'parsed' or 'error')."""
    query = """
    UPDATE documents
    SET status = %s
    WHERE doc_id = %s;
    """
    with conn.cursor() as cur:
        try:
            # doc_id is cast to str for the query execution
            cur.execute(query, (new_status, str(doc_id))) 
            conn.commit()
            print(f"[DB] Updated Document {doc_id} status to '{new_status}'")
        except psycopg2.Error as e:
            print(f"[DB ERROR] Could not update document status: {e}", file=sys.stderr)
            conn.rollback()


# ====================================================================
# II. STORAGE UTILITIES (Steps 1 & 2)
# Handles file I/O and classification before parsing.
# ====================================================================

def save_file_to_bucket(file_data: bytes, user_id: str, original_filename: str) -> str:
    """
    Saves the file data to the user's raw bucket directory (Simulated Bucket - Step 1).
    """
    # Define the user's specific raw file path: bucket/<user_id>/raw/<filename>
    user_raw_dir = BUCKET_BASE_DIR / user_id / "raw"
    
    # Create directory if it doesn't exist
    os.makedirs(user_raw_dir, exist_ok=True)
    
    # Sanitize the filename
    safe_filename = Path(original_filename).name
    final_file_path = user_raw_dir / safe_filename
    
    # Save the file content
    with open(final_file_path, 'wb') as f:
        f.write(file_data)
        
    # Return the full path to be stored in the 'file_path' DB column
    return str(final_file_path)


def detect_doc_type(original_filename: str) -> str:
    """
    Determines the document type based on the file extension (Step 2).
    Maps to doc_type_enum: ('bank_statement', 'bill', 'receipt', 'other')
    """
    file_path = Path(original_filename)
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        # PDFs are the primary format for multi-page bank statements
        return 'bank_statement'
    elif suffix in ('.jpg', '.jpeg', '.png', '.tiff'):
        # Image formats are usually bills or receipts for OCR
        return 'receipt' 
    else:
        # Default for all others, including Excel/CSV
        return 'other'


# ====================================================================
# III. PARSING & CLEANING (Day 2 Core Logic)
# Handles reading content from PDF and normalizing transaction data.
# ====================================================================


# Optional: Toggle between real parser and mock data for testing
USE_REAL_PARSER = os.getenv("USE_REAL_PARSER", "false").lower() == "true"

import pdfplumber
from parsers.bob import parse_bob
from parsers.pnb import parse_pnb
from parsers.sbi import parse_sbi
from parsers.federal import parse_federal_bank


def parse_statement(filepath: str):
    """
    Detects the bank from the PDF's first page and routes to the correct parser.
    Expected return:
        bank_name: str
        txns: list[{ date, description, amount }]
    """
    try:
        with pdfplumber.open(filepath) as pdf:
            first_text = pdf.pages[0].extract_text() or ""
            lower = first_text.lower()

            # -------------- BANK DETECTION LOGIC --------------
            if "bank of baroda" in lower or "statement of account" in lower:
                bank = "BOB"
                txns = parse_bob(pdf, filepath)

            elif "punjab national bank" in lower or "pnb" in lower:
                bank = "PNB"
                txns = parse_pnb(pdf, filepath)

            elif "state bank of india" in lower or "sbi" in lower:
                bank = "SBI"
                txns = parse_sbi(pdf, filepath)

            elif "federal bank" in lower:
                bank = "Federal Bank"
                txns = parse_federal_bank(pdf, filepath)

            else:
                print(f"[WARN] Could not detect bank for: {filepath}")
                return None, []
            
            # --------- ENSURE OUTPUT FORMAT IS CONSISTENT ---------
            normalized = []
            for row in txns:
                # Your individual parsers might return either:
                #  { date, description, amount } OR
                #  { date, description, debit, credit }
                if "amount" in row:
                    amt = row["amount"]
                else:
                    debit = float(row.get("debit", 0) or 0)
                    credit = float(row.get("credit", 0) or 0)
                    amt = credit - debit  # positive credit, negative debit

                normalized.append({
                    "date": row.get("date", ""),
                    "description": row.get("description", ""),
                    "amount": amt,
                })

            print(f"[PARSER] Parsed {len(normalized)} rows from {bank}.")
            return bank, normalized

    except Exception as e:
        print(f"[ERROR] PDF parsing failed for {filepath}: {e}")
        return None, []

def clean_amount(value):
    """
    Convert amount strings like '1,234.56', '(123.45)', '- 123' into float.
    (From normalize.py)
    """
    if value is None:
        return 0.0

    val = str(value).strip()

    # Handle negative amounts in parentheses (e.g., common in bank statements)
    if val.startswith("(") and val.endswith(")"):
        val = "-" + val[1:-1]

    # Remove commas & spaces
    val = val.replace(",", "").replace(" ", "")

    try:
        return float(val)
    except:
        return 0.0


def parse_date(raw_date):
    """Attempts to parse common date formats (From normalize.py)."""
    raw_date = raw_date.strip()

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            # Returns a date object
            return datetime.strptime(raw_date, fmt).date()
        except ValueError:
            pass

    raise ValueError(f"Unrecognized date format: {raw_date!r}")


def normalize_txn(tx: dict, statement_id: uuid.UUID, user_id: str):
    """
    Skips junk rows and normalizes raw transaction data to the canonical schema.
    """

    bad_values = (
        "accountholder", "account holder", "account number",
        "statement", "page", "opening balance", "closing balance",
        "debit", "credit"
    )

    raw_date = str(tx.get("date", "")).strip().lower()

    # Skip junk/header rows BEFORE date parsing
    if not raw_date or any(raw_date.startswith(b) for b in bad_values):
        return None

    # Safe date parsing
    try:
        txn_date = parse_date(tx["date"])
    except Exception:
        return None

    description = tx.get("description", "").strip()
    amount = clean_amount(tx.get("amount"))

    # Infer direction from sign
    if amount < 0:
        direction = "debit"
    elif amount > 0:
        direction = "credit"
    else:
        direction = None

    return {
        "user_id": user_id,
        "statement_id": statement_id,
        "txn_date": txn_date,
        "posting_date": txn_date,           # same as txn_date for now
        "description_raw": description,
        "description_clean": description,   # later you can clean this further
        "amount": amount,
        "direction": direction,
        "vendor": "UNCLASSIFIED",
        "category": "PENDING",
        "subcategory": None,
        "confidence": 0.0,
        "classification_source": None,
    }

# ====================================================================
# IV. MAIN PIPELINE EXECUTION (main_pipeline.py logic - Step 4)
# Orchestrates the entire process for one PDF file.
# ====================================================================

def process_pdf(conn, filepath: str, user_id):
    """
    Orchestrates the parsing, normalization, and DB insertion for a single PDF.
    """
    print(f"\nProcessing: {os.path.basename(filepath)}")
    
    doc_id = None # Used to track the document in case of failure

    try:
        # 1. PARSE STATEMENT: Extract raw data from PDF (using the stub)
        bank_name, raw_txns = parse_statement(filepath)
        
        if not bank_name or not raw_txns:
            print("[WARN] No transactions parsed or unknown bank. Skipping DB step.")
            return 0
    
        # 2. CREATE DOCUMENT & STATEMENT: Insert records into DB
        # The document is initially created here, fulfilling the Day 2 requirement for a successful flow.
        doc_id, statement_id = create_document_and_statement(
            conn,
            user_id=user_id,
            bank_name=bank_name,
            file_path=filepath, # Note: For this demo, filepath is the source file path, not the bucket path.
            original_filename=os.path.basename(filepath),
        )

        # 3. NORMALIZE TRANSACTIONS: Clean and format data
        normalized = []
        for tx in raw_txns:
            normalized_tx = normalize_txn(tx, statement_id=statement_id, user_id=user_id)
            if normalized_tx is not None:
                normalized.append(normalized_tx)

        # 4. INSERT INTO DB: Bulk insert the transactions
        insert_transactions(conn, normalized)
        print(f"[INFO] Inserted {len(normalized)} transactions into DB.")

        # 5. UPDATE DOCUMENT STATUS (SUCCESS)
        # Update documents.status from 'uploaded' to 'parsed'
        update_document_status(conn, doc_id, 'parsed') 
        
        return len(normalized)

    except Exception as e:
        print(f"[FATAL ERROR] Failed to process PDF {filepath}: {e}", file=sys.stderr)
        
        # 6. UPDATE DOCUMENT STATUS (FAILURE): Flag the document for review
        if doc_id:
            update_document_status(conn, doc_id, 'error') 
            
        return 0


def main(pdf_directory="./data/input"):
    """Main execution function to iterate over all PDFs in a directory."""
    if not DEFAULT_USER_ID:
        print(
            "[ERROR] DEFAULT_USER_ID not set. Check your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    pdf_dir = Path(pdf_directory)
    if not pdf_dir.is_dir():
        print(f"[ERROR] PDF directory does not exist: {pdf_dir}")
        # Create mock input directory for demo purposes
        os.makedirs(pdf_dir, exist_ok=True)
        print(f"[INFO] Created mock input directory: {pdf_dir}")
        # Create a mock PDF file content to process
        Path(pdf_dir / "sample_statement.pdf").write_bytes(b"%PDF-1.4...")


    pdf_files = sorted([str(p) for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])

    if not pdf_files:
        print(f"[WARN] No PDF files found in {pdf_dir}. Created a mock one.")
        pdf_files = [str(pdf_dir / "sample_statement.pdf")]

    print("\n====================================")
    print("SpendSight – Parsing & DB Ingestion")
    print("====================================\n")
    print(f"Found {len(pdf_files)} PDF file(s).")

    conn = get_db_connection()
    total_inserted = 0

    try:
        for f in pdf_files:
            # NOTE: For the Day 2 prototype, we use the local file path (f)
            # for 'file_path', simulating the document being ingested.
            total_inserted += process_pdf(conn, f, user_id=DEFAULT_USER_ID)
    finally:
        # Ensures the database connection is closed safely
        conn.close()

    print("\n------------------------------------")
    print(f"Total transactions inserted: {total_inserted}")
    print("------------------------------------\n")


if __name__ == "__main__":
    main()