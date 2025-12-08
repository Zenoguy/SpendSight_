import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

def upload_pdf_to_supabase(pdf_bytes: bytes, filename: str) -> str:
    """Upload PDF to Supabase Storage and return public URL"""
    unique_filename = f"{int(time.time())}_{filename}"
    
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{unique_filename}"
    
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/pdf"
    }
    
    response = requests.post(url, data=pdf_bytes, headers=headers)
    
    if response.status_code in [200, 201]:
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{unique_filename}"
        return public_url
    else:
        raise Exception(f"Supabase upload failed: {response.text}")
