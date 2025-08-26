from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from app.rag.schema import IngestRecord, ChunkRecord

# Deterministic, char-based chunking (no external deps)
DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 80

DEFAULT_IN_FILE = Path("artifacts/ingest/docs.jsonl")
DEFAULT_OUT_DIR = Path("artifacts/chunks")
DEFAULT_OUT_FILE = DEFAULT_OUT_DIR / "chunks.jsonl"


def chunk_text(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Tuple[int, int, str]]:
    # Return list of (start, end, slice) with fixed size and overlap; deterministic
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be 0 <= overlap < size")

    n = len(text)
    if n == 0:
        return []

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + size)
        slice_text = text[start:end]
        chunks.append((start, end, slice_text))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def iter_ingest(in_file: Path) -> Iterable[IngestRecord]:
    if not in_file.exists():
        return
    with in_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield IngestRecord.model_validate_json(line)


def run_chunker(
    in_file: Path = DEFAULT_IN_FILE,
    out_file: Path = DEFAULT_OUT_FILE,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as w:
        for rec in iter_ingest(in_file):
            pieces = chunk_text(rec.text, size=size, overlap=overlap)
            for idx, (s, e, txt) in enumerate(pieces):
                out = ChunkRecord(
                    doc_id=rec.doc_id,
                    rel_path=rec.rel_path,
                    page=rec.page,
                    chunk_id=idx,
                    span_start=s,
                    span_end=e,
                    text=txt,
                )
                w.write(json.dumps(out.model_dump(), ensure_ascii=False) + "\n")
    return out_file


def main():
    out = run_chunker(DEFAULT_IN_FILE, DEFAULT_OUT_FILE)
    print(str(out))


if __name__ == "__main__":
    main()
