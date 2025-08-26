from __future__ import annotations
import pytest

from app.rag.index_np import search_np
from app.rag.index_faiss import search_faiss


pytest_plugins = ("helpers",)


@pytest.mark.skipif(
    pytest.importorskip("faiss", reason="faiss not installed") is None,
    reason="faiss not installed",
)
def test_faiss_matches_np_topk_for_hash(corpus_hash):
    # parity check on top 5 doc+chunk ids
    # set 'allowed' dict here to prevent unpacking too many outputs from corpus_hash
    allowed = {"vectors_path", "meta_path", "backend", "dim", "k"}
    kw = {"k": 5, "backend": "hash", "dim": 32, **corpus_hash}
    kw = {k: v for k, v in kw.items() if k in allowed}

    q = "Quickbeam"
    hits_np = search_np(q, **kw)
    hits_faiss = search_faiss(q, **kw)

    # Compare by (doc_id, chunk_id) to avoid float tie issues
    np_ids = [(h.doc_id, h.chunk_id) for h in hits_np]
    faiss_ids = [(h.doc_id, h.chunk_id) for h in hits_faiss]
    assert np_ids == faiss_ids
