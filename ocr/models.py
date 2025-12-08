# models.py
from sqlalchemy import (
    Column, Integer, String, LargeBinary, DateTime, Text, func
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    original_filename = Column(String)
    mime_type = Column(String)
    extension = Column(String(10))
    data = Column(LargeBinary)  # original image bytes (bytea in Postgres)
    uploaded_at = Column(DateTime, server_default=func.now())

    ocr_text = Column(Text, nullable=True)       # final extracted text
    report_text = Column(Text, nullable=True)    # metadata report content
    pdf_data = Column(LargeBinary, nullable=True)  # generated PDF bytes
    table_data = Column(Text, nullable=True)     # extracted table data
    pdf_url = Column(String, nullable=True)      # Supabase PDF URL
