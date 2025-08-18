from __future__ import annotations
from typing import Iterable, List, Tuple
from pathlib import Path
import hashlib
import json
import numpy as np

from app.rag.schema import ChunkRecord


DEFAULT_CHUNKS = Path("artifacts/chunks/chunks.jsonl")
DEFAULT_EMB_DIR = Path("artifacts/emb")
DEFAULT_VECTORS = DEFAULT_EMB_DIR / "vectors.npy"
DEFAULT_META = DEFAULT_EMB_DIR / "meta.jsonl"


def _file_sha256(p: Path) -> str:
    """
    Py 3.11+ simple streaming hash.

    FIXME: if we ever need Python <3.11 or custom chunk sizes,
    switch to a manual loop:
        for block in iter(partial(f.read, CHUNK), b""):
            sha.update(block)

    This is to tune chunk size (e.g., 64 KiB / 1 MiB), add progress, or
    update multiple digests in one pass.

    Docs:
      - hashlib.file_digest: https://docs.python.org/3/library/hashlib.html#hashlib.file_digest
      - iter(callable, sentinel) block reader example: https://docs.python.org/3/library/functions.html#iter
      - io.DEFAULT_BUFFER_SIZE reference: https://docs.python.org/3/library/io.html#io.DEFAULT_BUFFER_SIZE
    """
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _iter_chunks(p: Path) -> Iterable[ChunkRecord]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ChunkRecord.model_validate_json(line)


def _hash_vec(texts: List[str], dim: int) -> np.ndarray:
    """Deterministic pseudo-embeddings in [-1,1), then L2-normalize."""
    out = np.empty((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        row = np.empty(dim, dtype=np.float32)
        for d in range(dim):
            h = hashlib.blake2s(f"{t}::{d}".encode(), digest_size=8).digest()
            val = int.from_bytes(h, "big", signed=False)
            row[d] = (val / 2**64) * 2.0 - 1.0
        out[i] = row
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out / norms


def run_embed_hash(
    chunks_path: Path = DEFAULT_CHUNKS,
    out_vectors: Path = DEFAULT_VECTORS,
    out_meta: Path = DEFAULT_META,
    dim: int = 384,
    batch_size: int = 64,
) -> Tuple[Path, Path]:
    texts: List[str] = []
    meta_rows: List[str] = []
    for idx, rec in enumerate(_iter_chunks(chunks_path)):
        texts.append(rec.text)
        meta_rows.append(
            json.dumps(
                {
                    "row": idx,
                    "doc_id": rec.doc_id,
                    "rel_path": rec.rel_path,
                    "page": rec.page,
                    "chunk_id": rec.chunk_id,
                    "span_start": rec.span_start,
                    "span_end": rec.span_end,
                },
                ensure_ascii=False,
            )
        )

    out_vectors.parent.mkdir(parents=True, exist_ok=True)
    if not texts:
        np.save(out_vectors, np.empty((0, dim), dtype=np.float32))
        out_meta.write_text("", encoding="utf-8")
        return out_vectors, out_meta

    # simple batch loop (hash backend is cheap, but keep the interface)
    vecs = []
    for i in range(0, len(texts), batch_size):
        vecs.append(_hash_vec(texts[i : i + batch_size], dim=dim))
    mat = np.vstack(vecs).astype(np.float32)
    np.save(out_vectors, mat)
    out_meta.write_text("\n".join(meta_rows) + "\n", encoding="utf-8")
    return out_vectors, out_meta
