from pathlib import Path
import json

from app.rag.ingest import run_ingest, DEFAULT_SRC


def test_ingest_writes_jsonl(tmp_path: Path, monkeypatch):
    # Use a temp artifacts directory but read from real fixtures
    custom_out = tmp_path / "ingest" / "docs.jsonl"
    out = run_ingest(DEFAULT_SRC, custom_out)

    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 3  # we added 3 .md fixtures in Commit 2

    # Validate one record shape
    sample = json.loads(lines[0])
    assert {
        "doc_id",
        "rel_path",
        "page",
        "span_start",
        "span_end",
        "text",
    } <= sample.keys()
    assert sample["page"] == 1
    assert isinstance(sample["doc_id"], str) and len(sample["doc_id"]) > 8
