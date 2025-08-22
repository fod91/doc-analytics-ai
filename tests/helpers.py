from __future__ import annotations
from pathlib import Path
import pytest

from app.rag.ingest import run_ingest, DEFAULT_SRC
from app.rag.chunk import run_chunker, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from app.rag.embed import run_embed


def build_corpus(tmp_dir: Path, *, backend: str, dim: int) -> dict:
    # Build an isolated, ephemeral corpus under tmp_dir and return artifact paths
    ingest_out = tmp_dir / "ingest.jsonl"
    chunks_out = tmp_dir / "chunks.jsonl"
    vectors = tmp_dir / "vectors.npy"
    meta = tmp_dir / "meta.jsonl"
    checksum = tmp_dir / "checksums.txt"

    run_ingest(DEFAULT_SRC, ingest_out)
    run_chunker(
        ingest_out, chunks_out, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP
    )
    run_embed(
        chunks_path=chunks_out,
        out_vectors=vectors,
        out_meta=meta,
        out_checksum=checksum,
        backend=backend,
        dim=dim,
    )

    return {
        "vectors_path": vectors,
        "meta_path": meta,
        "chunks_path": chunks_out,
        # tests will pass their own backend/dim here
    }


@pytest.fixture(scope="function")
def corpus_hash(tmp_path: Path):
    # Ephemeral artifacts for hash embeddings - use 32 dimensions for speed
    return build_corpus(tmp_path, backend="hash", dim=32)


@pytest.fixture(scope="function")
def corpus_st(tmp_path: Path):
    # Ephemeral artifacts for ST embeddings (384 dimensions for all-MiniLM-L6-v2). Skips if ST missing
    pytest.importorskip("sentence_transformers")
    return build_corpus(tmp_path, backend="st", dim=384)
