from app.rag.index_select import search as sel
import pytest


pytest_plugins = ("helpers",)

allowed = {"vectors_path", "meta_path", "backend", "dim", "k"}


def test_default_falls_back_to_np_when_faiss_missing(corpus_hash, monkeypatch):
    monkeypatch.setenv("FEATURE_GENAI", "1")
    monkeypatch.setenv("INDEX_BACKEND", "faiss")

    kws = dict(k=3, embed_backend="hash", dim=32, **corpus_hash)
    kws = {k: v for k, v in kws.items() if k in allowed}

    # simulate missing faiss
    monkeypatch.setitem(sel.__globals__, "search_faiss", None)
    hits = sel("Quickbeam", **kws)
    assert hits and all(h.rank >= 1 for h in hits)


@pytest.mark.skipif(
    pytest.importorskip("faiss", reason="faiss not installed") is None,
    reason="faiss not installed",
)
def test_default_uses_faiss_when_available(corpus_hash, monkeypatch):
    monkeypatch.setenv("FEATURE_GENAI", "1")
    monkeypatch.setenv("INDEX_BACKEND", "faiss")

    kws = dict(k=3, embed_backend="hash", dim=32, **corpus_hash)
    kws = {k: v for k, v in kws.items() if k in allowed}

    hits = sel("Quickbeam", **kws)
    assert hits and all(h.rank >= 1 for h in hits)
