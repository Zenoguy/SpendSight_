import os
import psycopg2
import socket
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# -----------------------------
# FORCE IPV4 (WORKS ON ALL PYTHON VERSIONS)
# -----------------------------
orig_getaddrinfo = socket.getaddrinfo

def getaddrinfo_ipv4_only(*args, **kwargs):
    results = orig_getaddrinfo(*args, **kwargs)
    return [r for r in results if r[0] == socket.AF_INET]  # keep IPv4 only

socket.getaddrinfo = getaddrinfo_ipv4_only
# -----------------------------

# Load environment variables
load_dotenv()

def test_supabase():
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        print("❌ Error: DATABASE_URL is not set in your .env file.")
        return

    print(f"Attempting to connect to: {db_url[:35]}...")

    try:
        parsed = urlparse(db_url)
        query = parse_qs(parsed.query)

        # Ensure SSL (required by Supabase)
        query["sslmode"] = ["require"]

        new_query = "&".join([f"{k}={v[0]}" for k, v in query.items()])

        final_url = (
            f"postgresql://{parsed.username}:{parsed.password}"
            f"@{parsed.hostname}:{parsed.port}{parsed.path}?{new_query}"
        )

        conn = psycopg2.connect(final_url)
        cursor = conn.cursor()

        cursor.execute("SELECT version();")
        version = cursor.fetchone()

        print("\n✅ SUCCESS! Connected to Supabase.")
        print(f"PostgreSQL Version: {version[0]}")

        cursor.close()
        conn.close()

    except Exception as e:
        print("\n❌ CONNECTION FAILED")
        print("Error details:", e)

if __name__ == "__main__":
    test_supabase()
