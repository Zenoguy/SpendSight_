from sqlalchemy import text
from db import engine

# Add table_data column to existing table
with engine.connect() as conn:
    conn.execute(text("ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS table_data TEXT"))
    conn.commit()
    print("Column 'table_data' added successfully!")
