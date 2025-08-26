from __future__ import annotations
from pydantic import BaseModel


class IngestRecord(BaseModel):
    doc_id: str
    rel_path: str
    page: int
    span_start: int
    span_end: int
    text: str


class ChunkRecord(BaseModel):
    doc_id: str
    rel_path: str
    page: int
    chunk_id: int
    span_start: int
    span_end: int
    text: str
