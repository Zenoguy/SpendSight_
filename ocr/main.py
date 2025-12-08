## main.py
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path
from datetime import datetime
import os
import json
import uuid

from db import SessionLocal, engine
from models import Base, UploadedFile
from ocr_models import OcrDoc
from ocr_utils import is_allowed_image, run_ocr_on_image_bytes, txt_to_pdf_bytes
from vercel_blob import upload_to_vercel_blob
from supabase_storage import upload_pdf_to_supabase
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from UnifiedPipeline import process_file
from PipeLine import get_db_connection

load_dotenv()
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")
DATABASE_URL = os.getenv("DATABASE_URL")

Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload")
async def upload_file(
    username: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    original_filename = file.filename
    ext = Path(original_filename).suffix.lower()

    if not is_allowed_image(original_filename):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    image_bytes = await file.read()

    # Upload image to Vercel Blob
    image_url = upload_to_vercel_blob(image_bytes, original_filename)

    # Run OCR
    ocr_text, extracted_tables = run_ocr_on_image_bytes(image_bytes, suffix=ext)

    # Generate PDF
    pdf_bytes = txt_to_pdf_bytes(ocr_text, username=username, tables=extracted_tables)
    
    # Upload PDF to Supabase
    pdf_filename = f"{username}_ocr.pdf"
    pdf_url = upload_pdf_to_supabase(pdf_bytes, pdf_filename)

    # Create JSON data
    json_data = json.dumps({"text": ocr_text})

    # Save to ocr_docs table
    ocr_doc = OcrDoc(
        user_id=uuid.UUID(DEFAULT_USER_ID),
        username=username,
        extracted_text=ocr_text,
        json_data=json_data,
        image_url=image_url
    )
    db.add(ocr_doc)
    db.commit()
    db.refresh(ocr_doc)

    # Save to uploaded_files table
    uploaded_file = UploadedFile(
        username=username,
        original_filename=original_filename,
        mime_type=file.content_type,
        extension=ext,
        data=image_bytes,
        ocr_text=ocr_text,
        pdf_data=pdf_bytes,
        pdf_url=pdf_url,
        report_text=f"Username: {username}\nFile: {original_filename}\nOCR ID: {ocr_doc.ocr_id}",
        table_data=json.dumps(extracted_tables) if extracted_tables else None
    )
    db.add(uploaded_file)
    db.commit()
    db.refresh(uploaded_file)

    # Call unified pipeline
    conn = get_db_connection()
    process_file(conn, pdf_url, "724d6378-9e58-427b-b3f1-72b4341e5ba3")
    conn.close()

    return {
        "ocr_id": str(ocr_doc.ocr_id),
        "file_id": uploaded_file.id,
        "user_id": str(ocr_doc.user_id),
        "username": username,
        "extracted_text": ocr_doc.extracted_text,
        "image_url": ocr_doc.image_url,
        "pdf_url": pdf_url,
        "created_at": ocr_doc.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "message": "OCR processed successfully",
    }

@app.get("/files/{file_id}")
def get_file_info(file_id: int, db: Session = Depends(get_db)):
    db_file = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    import json
    return {
        "id": db_file.id,
        "username": db_file.username,
        "original_filename": db_file.original_filename,
        "uploaded_at": db_file.uploaded_at.strftime("%Y-%m-%d %H:%M:%S"),
        "ocr_text": db_file.ocr_text,
        "report_text": db_file.report_text,
        "table_data": json.loads(db_file.table_data) if db_file.table_data else None,
    }

@app.get("/files/{file_id}/pdf")
def download_pdf(file_id: int, db: Session = Depends(get_db)):
    from fastapi.responses import Response
    
    db_file = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
    if not db_file or not db_file.pdf_data:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return Response(
        content=db_file.pdf_data,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={db_file.username}_ocr.pdf"}
    )

@app.post("/test-ocr")
async def test_ocr(file: UploadFile = File(...)):
    """Debug endpoint to test OCR output"""
    from ocr_utils import run_ocr_on_image_bytes
    image_bytes = await file.read()
    ext = Path(file.filename).suffix.lower()
    ocr_text, tables = run_ocr_on_image_bytes(image_bytes, suffix=ext)
    return {"ocr_text": ocr_text, "tables_found": len(tables), "length": len(ocr_text)}
