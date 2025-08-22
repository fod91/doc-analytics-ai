from __future__ import annotations
import pytest

from app.rag.retrieve import retrieve


pytest_plugins = ("helpers",)


def _contains_literal(q: str, s: str) -> bool:
    qn = (q or "").casefold().replace("-", " ")
    sn = (s or "").casefold().replace("-", " ")
    return qn in sn


@pytest.mark.parametrize("query", ["Quickbeam", "athelas", "barrow-blade"])
def test_keyword_rerank_improves_literal_top1_on_hash(corpus_hash, query):
    # A: baseline (no rerank)
    ctx_base = retrieve(
        query,
        k=3,
        dim=32,
        embed_backend="hash",
        rerank="none",
        candidate_multiplier=4,
        **corpus_hash,
    )
    # B: reranked
    ctx_rk = retrieve(
        query,
        k=3,
        dim=32,
        embed_backend="hash",
        rerank="keyword",
        candidate_multiplier=5,
        **corpus_hash,
    )

    assert ctx_base and ctx_rk, "no contexts returned"
    base_t1_has = _contains_literal(query, ctx_base[0].get("text", ""))
    rk_t1_has = _contains_literal(query, ctx_rk[0].get("text", ""))

    # If baseline already nails it, rerank should not degrade it, otherwise it should fix it
    if base_t1_has:
        assert rk_t1_has, f"rerank degraded literal match for {query}"
    else:
        assert rk_t1_has, f"rerank failed to surface literal top1 for {query}"


@pytest.mark.parametrize(
    "backend,dim,fixture_name,expected_auto",
    [
        ("hash", 32, "corpus_hash", "keyword"),
        ("st", 384, "corpus_st", "none"),
    ],
)
def test_auto_strategy_behaves_reasonably(
    request, backend, dim, fixture_name, expected_auto
):
    corpus = request.getfixturevalue(fixture_name)

    ctx_auto = retrieve(
        "Quickbeam",
        k=3,
        dim=dim,
        embed_backend=backend,
        rerank="auto",
        candidate_multiplier=5,
        **corpus,
    )
    ctx_ref = retrieve(
        "Quickbeam",
        k=3,
        dim=dim,
        embed_backend=backend,
        rerank=expected_auto,
        candidate_multiplier=5,
        **corpus,
    )

    # Compare top-1 doc+chunk id (not full text) to avoid brittle string asserts
    a = (ctx_auto[0]["doc_id"], ctx_auto[0]["chunk_id"])
    b = (ctx_ref[0]["doc_id"], ctx_ref[0]["chunk_id"])
    assert a == b
