import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io
import requests
from datetime import datetime

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

engine = create_engine(SUPABASE_DB_URL)

# Session storage for vector stores
vector_stores = {}

def get_user_ocr_docs(user_id: str):
    """Fetch all OCR docs for a user"""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT ocr_id, extracted_text, username FROM ocr_docs WHERE user_id = :user_id"),
            {"user_id": user_id}
        )
        return result.fetchall()

def create_vector_store(user_id: str):
    """Create vector store from user's OCR documents"""
    docs = get_user_ocr_docs(user_id)
    
    if not docs:
        return None
    
    # Format documents
    documents = []
    for doc in docs:
        ocr_id, extracted_text, username = doc
        documents.append(Document(
            page_content=extracted_text,
            metadata={"ocr_id": str(ocr_id), "username": username}
        ))
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings (using free HuggingFace)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Store in session
    session_id = str(uuid.uuid4())
    vector_stores[session_id] = vectorstore
    
    return session_id

def generate_pdf_report(content: str, title: str) -> bytes:
    """Generate PDF from text content"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 0.3*inch))
    
    for line in content.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['BodyText']))
            story.append(Spacer(1, 0.1*inch))
    
    doc.build(story)
    return buffer.getvalue()

def upload_to_supabase(pdf_data: bytes, filename: str) -> str:
    """Upload PDF to Supabase Storage"""
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/pdf"
    }
    response = requests.post(url, headers=headers, data=pdf_data)
    
    if response.status_code in [200, 201]:
        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
    return None

def save_report_to_db(user_id: str, query: str, report_content: str, pdf_url: str):
    """Save report metadata to reports table"""
    import json
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO reports (report_id, user_id, period, summary_json, insights, pdf_url, report_type, created_at)
                VALUES (:report_id, :user_id, :period, CAST(:summary_json AS jsonb), :insights, :pdf_url, :report_type, :created_at)
            """),
            {
                "report_id": str(uuid.uuid4()),
                "user_id": user_id,
                "period": datetime.utcnow().strftime("%Y-%m"),
                "summary_json": json.dumps({"query": query, "generated_by": "RAG"}),
                "insights": report_content,
                "pdf_url": pdf_url,
                "report_type": "RAG_GENERATED",
                "created_at": datetime.utcnow()
            }
        )
        conn.commit()

def query_rag(session_id: str, query: str, user_id: str = None):
    """Query RAG system and return top 3 contexts"""
    if session_id not in vector_stores:
        return {"error": "Session not found"}
    
    vectorstore = vector_stores[session_id]
    docs = vectorstore.similarity_search(query, k=3)
    
    # Check if query contains report keywords
    report_keywords = ['report', 'generate report', 'create report', 'detailed report', 'summary report']
    is_report_request = any(keyword in query.lower() for keyword in report_keywords)
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    if is_report_request:
        # Generate detailed report
        prompt = f"Context:\n{context}\n\nGenerate a comprehensive detailed report based on the context. Include analysis, insights, and key findings. Question: {query}"
        response = llm.invoke(prompt)
        answer = response.content
        
        # Generate PDF and upload
        pdf_data = generate_pdf_report(answer, f"Report: {query}")
        filename = f"report_{uuid.uuid4()}.pdf"
        pdf_url = upload_to_supabase(pdf_data, filename)
        
        if pdf_url and user_id:
            save_report_to_db(user_id, query, answer, pdf_url)
        
        return {
            "answer": answer,
            "contexts": [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs],
            "is_report": True,
            "pdf_url": pdf_url
        }
    else:
        # Normal conversation
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based only on the context provided:"
        response = llm.invoke(prompt)
        answer = response.content
        
        return {
            "answer": answer,
            "contexts": [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs],
            "is_report": False
        }
