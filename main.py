import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Depends,
    Query,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from psycopg2.pool import SimpleConnectionPool
import psycopg2

# Import unified pipeline + helpers
from UnifiedPipeline import (
    process_file,
    fetch_transactions_for_minilm,
    apply_minilm_to_txn,
)
# NOTE: PipeLine.py is still used by UnifiedPipeline internally for helpers
# (insert_transactions, insert_classification_log, etc.)

# -------------------------------------------------------
# ENV + GLOBALS
# -------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment")

if not DEFAULT_USER_ID:
    raise RuntimeError("DEFAULT_USER_ID is not set in environment")

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# psycopg2 connection pool (set in startup)
db_pool: Optional[SimpleConnectionPool] = None

# -------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------

app = FastAPI(title="SpendSight API", version="1.0.0")

# CORS – open for now; tighten later if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# DB DEPENDENCY (POOL)
# -------------------------------------------------------

@app.on_event("startup")
def on_startup():
    """
    Initialize PostgreSQL connection pool.
    """
    global db_pool
    # Tune minconn / maxconn based on server size
    db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=DATABASE_URL,
    )
    if not db_pool:
        raise RuntimeError("Failed to create database connection pool")


@app.on_event("shutdown")
def on_shutdown():
    """
    Cleanly close all DB connections.
    """
    global db_pool
    if db_pool:
        db_pool.closeall()
        db_pool = None


def get_db():
    """
    FastAPI dependency that yields a pooled DB connection.
    """
    if db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database pool is not initialized",
        )

    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)


# -------------------------------------------------------
# BACKGROUND TASK RUNNER
# -------------------------------------------------------

def run_pipeline_in_background(file_path: str, user_id: str):
    """
    Background task: run unified pipeline on a file.
    Uses its own pooled DB connection.
    """
    if db_pool is None:
        # No HTTPException here, this is background – just fail loudly in logs.
        raise RuntimeError("DB pool is not initialized in background task")

    conn = db_pool.getconn()
    try:
        processed, metrics = process_file(conn, file_path, user_id)
        # You can add logging here; for now, just print.
        print(
            f"[BG PIPELINE] file={file_path}, "
            f"transactions={processed}, metrics={metrics}"
        )
    except Exception as e:
        # Don't swallow errors silently; log them.
        print(f"[BG PIPELINE][ERROR] file={file_path}: {e}")
        # Optionally: mark document status=error here if you want.
        # That would require a doc_id; you'd then need to return/store it.
    finally:
        db_pool.putconn(conn)


# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------

@app.get("/", tags=["meta"])
def healthcheck():
    """
    Basic health endpoint.
    """
    return {"status": "ok", "service": "SpendSight API"}


# 1) UPLOAD ------------------------------------------------

@app.post("/documents/upload", tags=["documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF/image to local disk. We do NOT touch the DB here.
    Parsing + insertion happens in /documents/{file_id}/parse.
    """
    # Basic extension check – you can be stricter if needed
    original_name = file.filename or "upload"
    suffix = original_name.split(".")[-1].lower()

    if suffix not in ("pdf", "jpg", "jpeg", "png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Allowed: pdf, jpg, jpeg, png",
        )

    file_id = f"{uuid.uuid4()}.{suffix}"
    dest = UPLOAD_DIR / file_id

    try:
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        )

    return {
        "file_id": file_id,
        "original_filename": original_name,
        "path": str(dest),
    }


# 2) PARSE (INGEST + CLASSIFY) -----------------------------

@app.post("/documents/{file_id}/parse", tags=["documents"])
def parse_document(
    file_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Enqueue parsing + classification of the uploaded file.
    This calls UnifiedPipeline.process_file(conn, filepath, user_id).

    Returns immediately with 'queued' while work runs in background.
    """
    file_path = UPLOAD_DIR / file_id

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found. Upload first.",
        )

    # Fire-and-forget background processing
    background_tasks.add_task(
        run_pipeline_in_background,
        str(file_path),
        DEFAULT_USER_ID,
    )

    return {
        "status": "queued",
        "file_id": file_id,
        "path": str(file_path),
        "user_id": DEFAULT_USER_ID,
    }

@app.get("/documents/{file_id}/status", tags=["documents"])
def get_document_status(file_id: str, db=Depends(get_db)):
    q = """
    SELECT d.doc_id, d.status, s.statement_id
    FROM documents d
    LEFT JOIN statements s ON d.doc_id = s.doc_id
    WHERE d.file_path LIKE %s
    LIMIT 1;
    """
    with db.cursor() as cur:
        cur.execute(q, (f"%{file_id}%",))
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Not found")

    return {
        "doc_id": row[0],
        "status": row[1],
        "statement_id": row[2]
    }


# 3) CLASSIFY (MiniLM re-run) ------------------------------

@app.post("/statements/{statement_id}/classify", tags=["classification"])
def classify_statement_minilm(
    statement_id: str,
    db=Depends(get_db),
):
    """
    Re-run MiniLM classification for all transactions in a statement
    that still need semantic classification.

    Uses:
      - fetch_transactions_for_minilm(conn, statement_id)
      - apply_minilm_to_txn(conn, txn)
    """
    try:
        pending = fetch_transactions_for_minilm(db, statement_id)
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DB error: {e}",
        )

    attempted = len(pending)
    classified = 0
    still_pending = 0

    for txn in pending:
        label = apply_minilm_to_txn(db, txn)
        if label == "PENDING":
            still_pending += 1
        else:
            classified += 1

    return {
        "statement_id": statement_id,
        "mini_attempted": attempted,
        "mini_classified": classified,
        "mini_pending": still_pending,
    }


# 4) QUERY TRANSACTIONS ------------------------------------

@app.get("/transactions", tags=["query"])
def query_transactions(
    db=Depends(get_db),
    limit: int = Query(100, ge=1, le=1000),
    user_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    """
    Basic transaction query endpoint.

    Filters:
      - user_id (optional; if None, returns across all users)
      - category
      - vendor
      - date_from, date_to (transaction date range)
      - limit
    """
    clauses = []
    params = []

    if user_id:
        clauses.append("user_id = %s")
        params.append(user_id)

    if category:
        clauses.append("category = %s")
        params.append(category)

    if vendor:
        clauses.append("vendor ILIKE %s")
        params.append(f"%{vendor}%")

    if date_from:
        clauses.append("txn_date >= %s")
        params.append(date_from)

    if date_to:
        clauses.append("txn_date <= %s")
        params.append(date_to)

    where_clause = ""
    if clauses:
        where_clause = "WHERE " + " AND ".join(clauses)

    query = f"""
        SELECT
            txn_id,
            user_id,
            statement_id,
            txn_date,
            posting_date,
            description_raw,
            description_clean,
            amount,
            direction,
            vendor,
            category,
            subcategory,
            confidence,
            classification_source,
            created_at
        FROM transactions
        {where_clause}
        ORDER BY txn_date DESC, created_at DESC
        LIMIT %s;
    """

    params.append(limit)

    try:
        with db.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DB error: {e}",
        )

    # Transform rows → list of dicts
    results = [
        {colnames[i]: row[i] for i in range(len(colnames))}
        for row in rows
    ]

    return {
        "count": len(results),
        "results": results,
    }
