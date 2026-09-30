"""Database configuration and session management."""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

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


def get_db():
    """
    Dependency for getting a database session.
    Usage: 
        session = SessionLocal()
        try:
            # use session
        finally:
            session.close()
    
    Or in FastAPI:
        def get_db():
            db = SessionLocal()
            try:
                yield db
            finally:
                db.close()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database (create all tables).
    Call this from ingestion scripts or on first run.
    
    Note: For production, use Alembic migrations instead of create_all()
    """
    # Import all models so they're registered with Base
    from parking_assistant.db.models import (
        ParkingLocation, 
        PriceRule, 
        WorkingHours, 
        Availability,
        Reservation
    )
    
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized (all tables created)")
