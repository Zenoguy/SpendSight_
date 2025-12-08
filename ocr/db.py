# db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()  # loads .env into environment

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

if not DATABASE_URL:
    raise RuntimeError("SUPABASE_DB_URL is not set in environment variables")

# For Supabase, SSL is usually required. You already have ?sslmode=require in the URL.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # avoid stale connections
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
