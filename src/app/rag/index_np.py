from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import sys
import json
import numpy as np

from app.rag.embed import _embed_batch
from app.rag.embed import DEFAULT_VECTORS, DEFAULT_META


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    rel_path: str
    page: int
    chunk_id: int
    score: float
    rank: int


def _load_vectors(vectors_path: Path) -> np.ndarray:
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors file not found: {vectors_path}")
    mat = np.load(vectors_path)
    if mat.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape={mat.shape}")
    return mat.astype(np.float32, copy=False)


def _load_meta(meta_path: Path) -> List[Dict[str, Any]]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    rows: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Optional sanity: row index monotonic
    for i, m in enumerate(rows):
        if m.get("row") != i:
            # Still proceed; we can index by position
            break
    return rows


def search_np(
    query: str,
    k: int = 5,
    *,
    vectors_path: Path = DEFAULT_VECTORS,
    meta_path: Path = DEFAULT_META,
    backend: str = "hash",
    model: str = "unused",
    dim: int = 384,
) -> List[SearchResult]:
    """
    Computes cosine similarity (dot product) between the query embedding and
    precomputed normalized chunk embeddings in vectors.npy, returning top-k.
    Assumes row order in meta aligns with vectors rows

    # FIXME: note the dim here is hard-coded for all-MiniLM-L6-v2 for now
    """
    mat = _load_vectors(vectors_path)
    metas = _load_meta(meta_path)

    if mat.shape[0] == 0:
        return []

    # Ensure we embed query with the same backend/dim that created the matrix.
    q = _embed_batch([query], backend=backend, model=model, dim=dim)[0]  # [D]
    if q.shape[0] != mat.shape[1]:
        raise ValueError(f"query dim {q.shape[0]} != vectors dim {mat.shape[1]}")

    # cosine = dot product because both are normalized to unit length
    scores = mat @ q
    if k <= 0:
        return []

    k_eff = min(k, scores.shape[0])

    # argsort descending, use argpartition then local sort for speed on large N
    idxs = np.argpartition(scores, -k_eff)[-k_eff:]
    idxs = idxs[np.argsort(scores[idxs])[::-1]]

    results: List[SearchResult] = []
    for rank, i in enumerate(idxs, start=1):
        m = metas[i]
        results.append(
            SearchResult(
                doc_id=m["doc_id"],
                rel_path=m["rel_path"],
                page=int(m["page"]),
                chunk_id=int(m["chunk_id"]),
                score=float(scores[i]),
                rank=rank,
            )
        )
    return results


def main():
    # Simple CLI:
    #   python -m app.rag.index_np "your query" 5
    q = sys.argv[1] if len(sys.argv) > 1 else "test"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    hits = search_np(q, k=k)
    # | RANK | SCORE | DOC ID | rel_path#chunk_id |
    for h in hits:
        print(f"{h.rank:02d}  {h.score:+.4f}  {h.doc_id}  {h.rel_path}#{h.chunk_id}")


if __name__ == "__main__":
    main()
