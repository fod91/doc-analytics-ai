from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from .db import Base


class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(256), nullable=False)
    text = Column(String, nullable=False)
    label = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
