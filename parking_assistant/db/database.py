"""Database configuration and session management."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Database URL (SQLite file-based)
# Production would use: postgresql://user:pass@host/db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./parking_assistant.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False  # Set to True for SQL query logging during development
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def init_db():
    """
    Initialize database (create all tables).
    Call this from ingestion scripts or on first run.

    Note: For production, use Alembic migrations instead of create_all()
    """
    # Imported for the side effect: this registers every model on Base.metadata,
    # so create_all() below knows which tables to create.
    import parking_assistant.db.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized (all tables created)")
