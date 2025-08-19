from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from app.rag.index_select import search as index_search
from app.rag.embed import DEFAULT_META, DEFAULT_VECTORS
from app.rag.chunk import DEFAULT_OUT_FILE as DEFAULT_CHUNKS
from app.rag.rerank import rerank_keyword
from app.rag.config import RerankStrategy


def _load_chunks_map(chunks_path: Path) -> Dict[Tuple[str, int, int], str]:
    # Map (doc_id, page, chunk_id) -> text
    m: Dict[Tuple[str, int, int], str] = {}
    if not chunks_path.exists():
        return m
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m[(rec["doc_id"], int(rec["page"]), int(rec["chunk_id"]))] = rec["text"]
    return m


def retrieve(
    query: str,
    k: int = 5,
    *,
    index_backend: str | None = None,
    embed_backend: str | None = None,
    model: str | None = None,
    dim: int | None = None,
    vectors_path: Path = DEFAULT_VECTORS,
    meta_path: Path = DEFAULT_META,
    chunks_path: Path = DEFAULT_CHUNKS,
    rerank: RerankStrategy = RerankStrategy.auto,
    candidate_multiplier: int = 4,
):
    # Decide re-ranking strategy
    if rerank == RerankStrategy.auto:
        # heuristic: lexical rerank helps hash, rarely needed for ST
        strategy = (
            RerankStrategy.keyword
            if (embed_backend or "hash") == "hash"
            else RerankStrategy.none
        )
    else:
        strategy = rerank

    # Create a candidate
    # k0 = k * (candidate_multiplier if rerank else 1)
    k0 = k * (candidate_multiplier if strategy != RerankStrategy.none else 1)

    # Returns: contexts: [{doc_id, rel_path, page, chunk_id, score, rank, text}]
    hits = index_search(
        query,
        k=k0,
        index_backend=index_backend,
        embed_backend=embed_backend,
        model=model,
        dim=dim,
        vectors_path=vectors_path,
        meta_path=meta_path,
    )

    # Now build the contexts for each hit
    cmap = _load_chunks_map(chunks_path)
    contexts = []
    for h in hits:
        txt = cmap.get((h.doc_id, h.page, h.chunk_id), "")
        contexts.append(
            {
                "doc_id": h.doc_id,
                "rel_path": h.rel_path,
                "page": h.page,
                "chunk_id": h.chunk_id,
                "score": h.score,
                "rank": h.rank,
                "text": txt,
            }
        )

    # apply rerank strategy
    if strategy == RerankStrategy.keyword:
        contexts = rerank_keyword(query, contexts)[:k]
    else:
        contexts = contexts[:k]
    return contexts
