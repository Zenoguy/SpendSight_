import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

print("Connecting to:", os.getenv("DATABASE_URL"))

conn = psycopg2.connect(os.getenv("DATABASE_URL"), sslmode="require")
print("Connected OK!")
conn.close()
EOF
