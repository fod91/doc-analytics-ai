from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any

from app.rag.retrieve import retrieve
from app.rag.embed import DEFAULT_VECTORS, DEFAULT_META
from app.rag.chunk import DEFAULT_OUT_FILE as DEFAULT_CHUNKS


@dataclass
class Golden:
    query: str
    rel_path: Optional[str] = None
    literal: bool = False


def _norm(s: str) -> str:
    return (s or "").casefold().replace("-", " ")


def _contains(q: str, t: str) -> bool:
    return _norm(q) in _norm(t)


def load_goldens(p: Path) -> List[Golden]:
    rows: List[Golden] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        exp = obj.get("expect", {})
        rows.append(
            Golden(
                query=obj["query"],
                rel_path=exp.get("rel_path"),
                literal=bool(exp.get("literal", False)),
            )
        )
    return rows


def evaluate_goldens(
    goldens: Iterable[Golden],
    *,
    k: int = 3,
    embed_backend: str = "hash",
    dim: int = 32,
    rerank: str = "keyword",
    candidate_multiplier: int = 5,
    vectors_path: Path = DEFAULT_VECTORS,
    meta_path: Path = DEFAULT_META,
    chunks_path: Path = DEFAULT_CHUNKS,
) -> Dict[str, Any]:
    """
    Minimal retrieval evaluation for LOTR. Success is achieved for a query if:
      - any of the top-k contexts comes from expected rel_path (if provided)
      - AND if literal==True, the context text also contains the query

    Also compute Hit@1, Hit@3, and MRR based on first position that matches the above
    """
    n = 0
    h1 = h3 = 0
    mrr = 0.0
    per: Dict[str, Any] = {}

    for g in goldens:
        n += 1
        ctxs = retrieve(
            g.query,
            k=k,
            embed_backend=embed_backend,
            dim=dim,
            rerank=rerank,
            candidate_multiplier=candidate_multiplier,
            vectors_path=vectors_path,
            meta_path=meta_path,
            chunks_path=chunks_path,
        )

        # determine first position that satisfies expectations
        pos = None
        for i, c in enumerate(ctxs, start=1):
            rel_ok = (g.rel_path is None) or (c.get("rel_path") == g.rel_path)
            lit_ok = (not g.literal) or _contains(g.query, c.get("text", ""))
            if rel_ok and lit_ok:
                pos = i
                break

        per[g.query] = {
            "pos": pos,
            "topk": len(ctxs),
            "expect_rel_path": g.rel_path,
            "literal": g.literal,
            "top1_rel_path": ctxs[0]["rel_path"] if ctxs else None,
        }
        if pos == 1:
            h1 += 1
        if pos and pos <= 3:
            h3 += 1
        if pos:
            mrr += 1.0 / pos

    return {
        "n": n,
        "hit_at_1": h1,
        "hit_at_3": h3,
        "mrr": (mrr / n) if n else 0.0,
        "per_query": per,
    }


if __name__ == "__main__":
    goldens_path = Path("evals/golden_lotr.jsonl")
    if not goldens_path.exists():
        print(
            json.dumps({"error": "goldens file not found", "path": str(goldens_path)})
        )
        exit(1)

    gs = load_goldens(goldens_path)
    out = evaluate_goldens(gs)
    print(json.dumps(out, indent=2))
