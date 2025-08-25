from __future__ import annotations
from pathlib import Path
from app.rag.eval_small import load_goldens, evaluate_goldens

pytest_plugins = ("helpers",)


def test_eval_lite_runs_and_scores(corpus_hash, tmp_path: Path):
    src = Path("evals/golden_lotr.jsonl")
    assert src.exists(), "golden LOTR file missing"
    dst = tmp_path / "golden_lotr.jsonl"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # vectors/meta/chunks from ephemeral fixture (corpus_hash)
    goldens = load_goldens(dst)
    res = evaluate_goldens(
        goldens,
        k=3,
        embed_backend="hash",
        dim=32,
        rerank="keyword",
        candidate_multiplier=5,
        **corpus_hash,
    )
    assert set(res) >= {"n", "hit_at_1", "hit_at_3", "mrr", "per_query"}
    assert res["n"] >= 3  # at least a few queries present
    # ensure we recorded "per-query" positions
    for g in goldens:
        assert g.query in res["per_query"]
