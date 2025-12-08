-- =========================================
-- Extensions (PostgreSQL)
-- =========================================

-- UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Vector extension (optional, for embeddings / RAG)
-- Comment this out if your environment doesn't support it yet.
CREATE EXTENSION IF NOT EXISTS vector;

-- =========================================
-- ENUM Types
-- =========================================

CREATE TYPE doc_type_enum AS ENUM ('bank_statement', 'bill', 'receipt', 'other');

CREATE TYPE document_status_enum AS ENUM ('uploaded', 'parsed', 'error');

CREATE TYPE statement_status_enum AS ENUM ('parsed', 'partial', 'error');

CREATE TYPE direction_enum AS ENUM ('debit', 'credit');

CREATE TYPE classification_source_enum AS ENUM ('regex', 'bert', 'llm');

CREATE TYPE stage_enum AS ENUM ('regex', 'bert', 'llm');

-- =========================================
-- USERS
-- =========================================

CREATE TABLE users (
    user_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    bucket_path   TEXT,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tier          TEXT DEFAULT 'free'

);

-- =========================================
-- DOCUMENTS
-- =========================================

CREATE TABLE documents (
    doc_id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id           UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    doc_type          doc_type_enum NOT NULL,
    file_path         TEXT NOT NULL,
    original_filename TEXT,
    upload_time       TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status            document_status_enum NOT NULL DEFAULT 'uploaded',
    raw_text          TEXT,
    meta              JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_doc_type ON documents(doc_type);
CREATE INDEX idx_documents_status ON documents(status);

-- =========================================
-- STATEMENTS (bank statements only)
-- =========================================

CREATE TABLE statements (
    statement_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id            UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    user_id           UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    period_start      DATE,
    period_end        DATE,
    account_number    TEXT,
    bank_name         TEXT,
    status            statement_status_enum NOT NULL DEFAULT 'parsed',
    parsed_table_path TEXT -- optional CSV/Excel path in bucket
);

CREATE INDEX idx_statements_user_id ON statements(user_id);
CREATE INDEX idx_statements_period ON statements(user_id, period_start, period_end);

-- =========================================
-- TRANSACTIONS (core table)
-- =========================================

CREATE TABLE transactions (
    txn_id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    statement_id         UUID REFERENCES statements(statement_id) ON DELETE CASCADE,
    user_id              UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    txn_date             DATE NOT NULL,
    posting_date         DATE,
    description_raw      TEXT NOT NULL,
    description_clean    TEXT,
    amount               NUMERIC(12,2) NOT NULL,
    direction            direction_enum,
    vendor               TEXT,
    category             TEXT,
    subcategory          TEXT,
    confidence           REAL,
    classification_source classification_source_enum,
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Important indexes for analytics and queries
CREATE INDEX idx_transactions_user_date ON transactions(user_id, txn_date);
CREATE INDEX idx_transactions_vendor ON transactions(vendor);
CREATE INDEX idx_transactions_category ON transactions(category);

-- =========================================
-- CLASSIFICATION_LOG
-- =========================================

CREATE TABLE classification_log (
    log_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    txn_id     UUID NOT NULL REFERENCES transactions(txn_id) ON DELETE CASCADE,
    stage      stage_enum NOT NULL,
    prediction TEXT,
    confidence REAL,
    timestamp  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    meta       JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_classlog_txn_id ON classification_log(txn_id);
CREATE INDEX idx_classlog_stage ON classification_log(stage);

-- =========================================
-- OCR_DOCS (bills, receipts, etc.)
-- =========================================

CREATE TABLE ocr_docs (
    ocr_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id         UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    user_id        UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    extracted_text TEXT,
    vendor         TEXT,
    invoice_date   DATE,
    total_amount   NUMERIC(12,2),
    json_data      JSONB DEFAULT '{}'::jsonb,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_ocr_user_id ON ocr_docs(user_id);
CREATE INDEX idx_ocr_vendor ON ocr_docs(vendor);

-- =========================================
-- EMBEDDINGS (for RAG)
-- =========================================

CREATE TABLE embeddings (
    emb_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    doc_id      UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_id    TEXT NOT NULL,
    embedding   vector(1536), -- adjust dimension if needed
    chunk_text  TEXT,
    meta        JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_embeddings_user_id ON embeddings(user_id);
CREATE INDEX idx_embeddings_doc_id ON embeddings(doc_id);

-- Optional: vector index (if you use pgvector)
-- CREATE INDEX idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_l2_ops)
-- WITH (lists = 100);

-- =========================================
-- REPORTS (monthly/yearly)
-- =========================================

CREATE TABLE reports (
    report_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id       UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    period        TEXT NOT NULL, -- e.g., '2024-01', '2024-Q1'
    summary_json  JSONB DEFAULT '{}'::jsonb,
    insights      TEXT,
    file_path     TEXT,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_reports_user_period ON reports(user_id, period);
