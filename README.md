# SpendSight: Automated Financial Document Intelligence

A Hybrid Rule-Based + ML + LLM System for Transaction Extraction, Categorization & Insights

SpendSight is an end-to-end financial document processing system that parses bank statements and OCR-extracted PDFs, normalizes transactions, classifies them using a hybrid pipeline (Regex → Heuristics → MiniLM → LLM fallback), stores them securely in a relational database, and generates AI-driven dashboards and user-query insights using RAG.

This repository contains all components required for ingestion, classification, analytics, and reporting.

---

## 🚀 Key Features

### *1. Multi-Source Document Input*

* Accepts *PDFs, **scanned images, and **photos of receipts/statements*.
* OCR auto-processing for images using *Table Transformer* + *LaTr-style recognition*.

### *2. Unified Parsing Pipeline*

* Bank-specific parsers (BOB, PNB, SBI, Federal Bank)
* Generic OCR parser for image-derived PDFs
* Automatic bank detection + fallback handling

### *3. Hybrid AI Classification Architecture*

1. *Regex Engine* (fast deterministic rules)
2. *Heuristic Classifier* (smart keyword patterns)
3. *MiniLM/BERT Classifier* (local semantic model)
4. *LLM Fallback* (Gemini-based batch classification)

### *4. Secure Storage & Data Model*

* documents, statements, transactions
* Per-user isolation
* Optional Supabase storage for PDFs
* Audit-friendly classification logs

### *5. Dashboard Analytics*

* Category spending distribution
* Monthly spending
* Vendor analytics
* Summary statistics
* Auto-snapshots stored in reports table

### *6. RAG-Powered User Insights (Query Engine)*

* Per-user vector store embeddings
* Natural-language Q&A over transactions + summaries
  
---

# ⚙ Installation

### *1. Clone the Project*

bash
git clone https://github.com/yourrepo/spendsight.git
cd SpendSight


### *2. Create a Virtual Environment*

bash
python3 -m venv venv
source venv/bin/activate


### *3. Install Requirements*

bash
pip install -r requirements.txt


### *4. Environment Variables*

Create a .env file:


DATABASE_URL=your_supabase_postgres_url
DEFAULT_USER_ID=the_uuid_user
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
SUPABASE_BUCKET_NAME=test
BLOB_READ_WRITE_TOKEN=your_vercel_blob_token


---

# 📄 OCR Module Setup

Start OCR server:

bash
cd ocr
python3 main.py


This provides:

* POST /upload → Upload image/PDF, run OCR, store document, trigger UnifiedPipeline automatically
* GET /files/{id} → Retrieve OCR metadata
* GET /files/{id}/pdf → Download generated PDF

The OCR module:

✔ Uploads file to Vercel Blob
✔ Runs OCR + table extraction
✔ Generates a cleaned PDF for ingestion
✔ Saves metadata into SQL tables
✔ Writes the final PDF into data/input/
✔ Calls UnifiedPipeline.process_file() automatically

---

# 🔄 Unified Ingestion Pipeline

To run the full parsing + classification pipeline on all PDFs in data/input/:

bash
python3 UnifiedPipeline.py


The pipeline will:

1. Detect bank or use the generic OCR parser
2. Extract raw transactions
3. Normalize structure
4. Insert into DB
5. Run Regex → Heuristics → MiniLM → LLM
6. Store classification logs
7. Save dashboard snapshot

On completion, it prints a full breakdown:


Total transactions inserted: 229
Regex: attempted=229, classified=180
Heuristics: attempted=49, classified=22
MiniLM: attempted=27, classified=21
LLM: attempted=6, classified=6


---

# 📊 Dashboard Analytics

Generate charts visualizing pipeline performance:

bash
python3 pipeline_visuals.py


Produces:

* Workload distribution (Regex vs MiniLM vs LLM)
* Classification funnel
* ROI plot (volume vs strong confidence)
* Pending over time drift chart

Outputs are saved to:


data/reports/pipeline_metrics/


---

# 💬 RAG Insights Engine (Optional)

Inside the rag/ module (if present), the system:

1. Embeds user transactions
2. Stores vectors per user
3. Answers voice/text questions like:

   * “Why was my spending high last month?”
   * “Show me all grocery purchases above ₹500.”
   * “Summarize my October financial behaviour.”

Uses hybrid SQL + retrieval-augmented generation.

---

# 🔐 Security & Compliance

SpendSight is designed with financial-grade constraints:

* TLS 1.2+
* Per-user DB + storage isolation
* No cross-user vector leakage
* Explicit admin access controls
* GDPR/CCPA delete-on-request compatibility
* PCI-DSS-compliant handling of sensitive financial identifiers

---

# 🛠 Known Limitations

* OCR parser accuracy varies for poor-quality scanned images
* Bank formats outside India require new parsers
* MiniLM classifier may mis-categorize rare merchant names
* LLM fallback is slow and expensive → used sparingly

---

# 🧭 Roadmap

* Incremental learning pipeline for MiniLM
* New bank parsers (ICICI, HDFC, Axis)
* Real-time user alert engine
* Interactive dashboard with drill-downs

---

# 👥 Credits

* Parsing & Classification Pipeline — Shreyan Ghosh & Sreedeep Dey.
* OCR Subsystem and RAG— Sambhranta Ghosh.
* Dashboard & Report Generation — Shreyan Ghosh.
* Architecture & Integration — Shreyan Ghosh & Arka Ghosh.

---
