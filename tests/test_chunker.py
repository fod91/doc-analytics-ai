from __future__ import annotations

import json
from pathlib import Path

from app.rag.ingest import run_ingest, DEFAULT_SRC
from app.rag.chunk import run_chunker, chunk_text, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP


def read_jsonl(p: Path):
    return [
        json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()
    ]


def test_chunk_text_deterministic_and_overlap():
    text = "abcdefghijklmnopqrstuvwxyz" * 50  # 1300 chars
    chunks1 = chunk_text(text, size=200, overlap=25)
    chunks2 = chunk_text(text, size=200, overlap=25)
    assert [(s, e) for s, e, _ in chunks1] == [(s, e) for s, e, _ in chunks2]
    # Check overlap property (except first)
    for i in range(1, len(chunks1)):
        prev_s, prev_e, _ = chunks1[i - 1]
        cur_s, cur_e, _ = chunks1[i]
        assert cur_s == max(0, prev_e - 25)
        assert cur_e - cur_s <= 200


def test_chunker_pipeline_from_ingest(tmp_path: Path, monkeypatch):
    # 1) run ingest into temp file (uses real fixtures)
    ingest_out = tmp_path / "ingest.jsonl"
    run_ingest(DEFAULT_SRC, ingest_out)

    # Ensure we have at least the 3 LOTR fixtures
    lines = read_jsonl(ingest_out)
    assert len(lines) >= 3

    # 2) run chunker twice and compare determinism on spans & IDs
    chunks_out1 = tmp_path / "chunks1.jsonl"
    chunks_out2 = tmp_path / "chunks2.jsonl"

    run_chunker(
        ingest_out, chunks_out1, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP
    )
    run_chunker(
        ingest_out, chunks_out2, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP
    )

    c1 = read_jsonl(chunks_out1)
    c2 = read_jsonl(chunks_out2)

    assert len(c1) == len(c2) >= len(lines)  # at least one chunk per doc
    # determinism: for each record, (doc_id, rel_path, page, chunk_id, spans) match
    for a, b in zip(c1, c2):
        assert (a["doc_id"], a["rel_path"], a["page"], a["chunk_id"]) == (
            b["doc_id"],
            b["rel_path"],
            b["page"],
            b["chunk_id"],
        )
        assert (a["span_start"], a["span_end"]) == (b["span_start"], b["span_end"])
        assert isinstance(a["text"], str) and isinstance(b["text"], str)
