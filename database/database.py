"""
Stage 1 (part): Storage strategy
------------------------------------------------------------------
Uses SQLite for local development/demo (zero setup, works instantly
for the hackathon prototype). For the real deployment, swap
DATABASE_URL for a Postgres/PostGIS URL, e.g.:

    postgresql://user:password@host:5432/gujarat_police_db

Nothing else in the codebase needs to change — SQLAlchemy handles
the dialect differences.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./cham.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
