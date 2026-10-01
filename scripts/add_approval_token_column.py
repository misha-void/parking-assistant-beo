"""One-off migration: add approval_token column to reservation table (Stage 2)."""

import sqlite3

con = sqlite3.connect("parking_assistant.db")
cur = con.cursor()

cols = [row[1] for row in cur.execute("PRAGMA table_info(reservation)").fetchall()]

if "approval_token" not in cols:
    cur.execute("ALTER TABLE reservation ADD COLUMN approval_token VARCHAR(64)")
    con.commit()
    print("✅ approval_token column added.")
else:
    print("ℹ️ approval_token column already exists.")

con.close()
