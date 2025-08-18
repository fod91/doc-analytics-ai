from pathlib import Path
import numpy as np
from app.rag.ingest import run_ingest, DEFAULT_SRC
from app.rag.chunk import run_chunker, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from app.rag.embed import run_embed_hash


def test_run_embed_hash_creates_outputs(tmp_path: Path):
    ingest_out = tmp_path / "ingest.jsonl"
    run_ingest(DEFAULT_SRC, ingest_out)
    chunks_out = tmp_path / "chunks.jsonl"
    run_chunker(
        ingest_out, chunks_out, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP
    )

    vectors_path = tmp_path / "vectors.npy"
    meta_path = tmp_path / "meta.jsonl"
    v, m = run_embed_hash(
        chunks_out, vectors_path, meta_path, dim=32, batch_size=16, backend="hash"
    )

    assert v.exists() and m.exists()
    mat = np.load(v)
    assert mat.ndim == 2 and mat.shape[1] == 32 and mat.shape[0] >= 3

    # determinism
    v2, _ = run_embed_hash(
        chunks_out, vectors_path, meta_path, dim=32, batch_size=16, backend="hash"
    )
    mat2 = np.load(v2)
    assert np.allclose(mat, mat2)
