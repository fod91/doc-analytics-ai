from sqlalchemy import Column, Integer, String, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from .db import Base


class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(256), nullable=False)
    text = Column(String, nullable=False)
    label = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ObjectStoreItem(Base):
    __tablename__ = "object_store_items"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bucket = Column(String, nullable=False)
    key = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    etag = Column(String, nullable=True)
    source = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    __table_args__ = (UniqueConstraint("bucket", "key", name="uq_bucket_key"),)
