from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import json

from app.rag.ingest import run_ingest, DEFAULT_SRC
from app.rag.chunk import run_chunker, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from app.rag.embed import run_embed
from app.rag.index_np import search_np, _load_vectors, _load_meta


def write_meta(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_np_missing_files(tmp_path: Path):
    v = tmp_path / "no_vectors.npy"
    m = tmp_path / "no_meta.jsonl"
    with pytest.raises(FileNotFoundError):
        _load_vectors(v)
    with pytest.raises(FileNotFoundError):
        _load_meta(m)


def test_np_empty_vectors_returns_empty(tmp_path: Path):
    v = tmp_path / "vectors.npy"
    m = tmp_path / "meta.jsonl"
    # 0 rows, 16 dims
    np.save(v, np.empty((0, 16), dtype=np.float32))
    write_meta(m, [])  # no rows
    hits = search_np(
        "anything", k=5, vectors_path=v, meta_path=m, backend="hash", dim=16
    )
    assert hits == []


def test_np_dim_mismatch_raises(tmp_path: Path):
    v = tmp_path / "vectors.npy"
    m = tmp_path / "meta.jsonl"
    np.save(v, np.zeros((3, 16), dtype=np.float32))
    write_meta(
        m,
        [
            {
                "row": i,
                "doc_id": "d",
                "rel_path": "x",
                "page": 1,
                "chunk_id": i,
                "span_start": 0,
                "span_end": 1,
            }
            for i in range(3)
        ],
    )
    with pytest.raises(ValueError):
        search_np("q", k=2, vectors_path=v, meta_path=m, backend="hash", dim=32)


def test_np_k_zero_returns_empty(tmp_path: Path):
    v = tmp_path / "vectors.npy"
    m = tmp_path / "meta.jsonl"
    np.save(v, np.zeros((3, 8), dtype=np.float32))
    write_meta(
        m,
        [
            {
                "row": i,
                "doc_id": "d",
                "rel_path": "x",
                "page": 1,
                "chunk_id": i,
                "span_start": 0,
                "span_end": 1,
            }
            for i in range(3)
        ],
    )
    hits = search_np("q", k=0, vectors_path=v, meta_path=m, backend="hash", dim=8)
    assert hits == []


def test_index_np_search_interface_and_determinism(tmp_path: Path):
    # Pipeline: ingest -> chunk -> embed; into tmp artifacts
    ingest_out = tmp_path / "ingest.jsonl"
    run_ingest(DEFAULT_SRC, ingest_out)

    chunks_out = tmp_path / "chunks.jsonl"
    run_chunker(
        ingest_out, chunks_out, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP
    )

    vectors_path = tmp_path / "vectors.npy"
    meta_path = tmp_path / "meta.jsonl"

    run_embed(
        chunks_path=chunks_out,
        out_vectors=vectors_path,
        out_meta=meta_path,
        backend="hash",
        dim=32,
        batch_size=16,
    )

    # Search for any term (hash backend isn't semantic, we test shape and determinism)
    q = "rowans"  # from lots_twotowers.md
    hits1 = search_np(
        q, k=5, vectors_path=vectors_path, meta_path=meta_path, backend="hash", dim=32
    )
    hits2 = search_np(
        q, k=5, vectors_path=vectors_path, meta_path=meta_path, backend="hash", dim=32
    )
    assert len(hits1) == len(hits2) == min(5, len(np.load(vectors_path)))

    # Deterministic order and scores
    rows1 = [(h.doc_id, h.chunk_id, round(h.score, 6)) for h in hits1]
    rows2 = [(h.doc_id, h.chunk_id, round(h.score, 6)) for h in hits2]
    assert rows1 == rows2

    # Basic fields present and rank monotonic
    for i, h in enumerate(hits1, start=1):
        assert h.rank == i
        assert isinstance(h.doc_id, str)
        assert 0 <= i <= 5
