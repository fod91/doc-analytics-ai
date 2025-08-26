from pathlib import Path
import numpy as np
from app.rag.ingest import run_ingest, DEFAULT_SRC
from app.rag.chunk import run_chunker, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from app.rag.embed import run_embed


def test_run_embed_creates_outputs(tmp_path: Path):
    ingest_out = tmp_path / "ingest.jsonl"
    run_ingest(DEFAULT_SRC, ingest_out)
    chunks_out = tmp_path / "chunks.jsonl"
    run_chunker(
        ingest_out, chunks_out, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP
    )

    vectors_path = tmp_path / "vectors.npy"
    meta_path = tmp_path / "meta.jsonl"
    chk = tmp_path / "checksums.txt"

    run_embed(
        chunks_out,
        vectors_path,
        meta_path,
        out_checksum=chk,
        backend="hash",
        dim=16,
        batch_size=8,
    )
    mtime1 = vectors_path.stat().st_mtime
    # second run should be a cache hit: same fingerprint -> files reused
    run_embed(
        chunks_out,
        vectors_path,
        meta_path,
        out_checksum=chk,
        backend="hash",
        dim=16,
        batch_size=8,
    )
    mtime2 = vectors_path.stat().st_mtime

    assert chk.exists() and "backend=hash" in chk.read_text(encoding="utf-8")
    assert np.load(vectors_path).shape[1] == 16
    # conservative check: file wasn't rewritten (mtime unchanged or near-equal)
    assert int(mtime1) == int(mtime2)
