import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()

BLOB_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")

def upload_to_vercel_blob(file_bytes: bytes, filename: str) -> str:
    """Upload file to Vercel Blob and return URL"""
    # Add timestamp to make filename unique
    unique_filename = f"{int(time.time())}_{filename}"
    url = f"https://blob.vercel-storage.com/{unique_filename}"
    
    headers = {
        "Authorization": f"Bearer {BLOB_TOKEN}",
        "x-content-type": "image/*"
    }
    
    response = requests.put(url, data=file_bytes, headers=headers)
    
    if response.status_code in [200, 201]:
        return response.json().get("url")
    else:
        raise Exception(f"Vercel Blob upload failed: {response.text}")
