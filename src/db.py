# src/db.py
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    JSON,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session


Base = declarative_base()
_engine = None
SessionLocal: sessionmaker[Session] | None = None


def init_db(database_url: str) -> None:
    global _engine, SessionLocal
    _engine = create_engine(database_url, future=True)
    SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)

    _ensure_schema(_engine)
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return SessionLocal()


def _ensure_schema(engine) -> None:
    """Lightweight compatibility shim until full migrations are added."""

    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as conn:
        info = conn.execute(text("PRAGMA table_info(people)"))
        columns = {row[1] for row in info.fetchall()}
        if "ignore" not in columns:
            conn.execute(text("ALTER TABLE people ADD COLUMN ignore BOOLEAN DEFAULT 0"))


class Person(Base):
    __tablename__ = "people"

    id = Column(Integer, primary_key=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=True)
    email = Column(String(255), nullable=True)
    vip = Column(Boolean, default=False)
    ignore = Column(Boolean, default=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    encodings = relationship("KnownFaceEncoding", back_populates="person")
    recognized_faces = relationship("RecognizedFace", back_populates="person")


class KnownFaceEncoding(Base):
    __tablename__ = "known_face_encodings"

    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("people.id"), nullable=False)
    image_path = Column(Text, nullable=True)
    encoding = Column(LargeBinary, nullable=False)
    encoding_dim = Column(Integer, nullable=False, default=128)
    distance_threshold = Column(Float, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    person = relationship("Person", back_populates="encodings")


class RecognizedFace(Base):
    __tablename__ = "recognized_faces"

    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("people.id"), nullable=False)
    image_path = Column(Text, nullable=False)
    attributes = Column(JSON, nullable=True)
    distance = Column(Float, nullable=True)
    detected_at = Column(DateTime, default=dt.datetime.utcnow)

    person = relationship("Person", back_populates="recognized_faces")


class UnknownFace(Base):
    __tablename__ = "unknown_faces"

    id = Column(Integer, primary_key=True)
    image_path = Column(Text, nullable=False)
    encoding = Column(LargeBinary, nullable=True)
    encoding_dim = Column(Integer, nullable=True)
    attributes = Column(JSON, nullable=True)
    detected_at = Column(DateTime, default=dt.datetime.utcnow)
