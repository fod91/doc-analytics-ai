from fastapi.testclient import TestClient

from app.rag.ingest import run_ingest, DEFAULT_SRC
from app.rag.chunk import run_chunker
from app.rag.embed import run_embed

# from doc_analytics_ai.api import app as fastapi_app
from doc_analytics_ai.api import create_app


def test_genai_ask_endpoint(monkeypatch):
    # Ensure feature on
    monkeypatch.setenv("FEATURE_GENAI", "1")

    # Build artifacts at default locations (simple for this commit)
    run_ingest(DEFAULT_SRC)  # default -> artifacts/ingest/docs.jsonl
    run_chunker()  # default -> artifacts/chunks/chunks.jsonl
    run_embed(dim=32)  # default -> artifacts/emb/* with 32-D for speed

    client = TestClient(create_app())
    r = client.get(
        "/genai/ask",
        params={"query": "Quickbeam", "k": 3, "embed_backend": "hash", "dim": 32},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data.get("answer"), str) and len(data["answer"]) > 0
    cits = data.get("citations") or []
    assert 1 <= len(cits) <= 3
    # sanity: each citation has fields
    assert {"rank", "score", "doc_id", "rel_path", "page", "chunk_id"} <= set(
        cits[0].keys()
    )
