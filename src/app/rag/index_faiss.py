from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np

try:
    import faiss
except Exception as e:
    raise RuntimeError(
        "faiss is not installed. Install faiss-cpu to use this backend."
    ) from e

from app.rag.embed import _embed_batch
from app.rag.index_np import _load_vectors, _load_meta
from app.rag.index_np import SearchResult


def _ensure_unit_norm(mat: np.ndarray) -> np.ndarray:
    # FAISS IndexFlatIP assumes inner product; cosine needs unit-norm
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    # If already ~1, skip extra work
    if np.allclose(norms, 1.0, atol=1e-3):
        return mat
    return mat / norms


def _build_flat_index(vecs: np.ndarray) -> "faiss.IndexFlatIP":
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # exact inner product (fast on small corpora)
    index.add(vecs.astype(np.float32, copy=False))
    return index


def search_faiss(
    query: str,
    k: int = 5,
    *,
    vectors_path: Path,
    meta_path: Path,
    backend: str = "hash",
    model: str = "unused",
    dim: int = 384,
) -> List[SearchResult]:
    # Cosine-search via FAISS (IndexFlatIP) over precomputed and aligned vectors
    mat = _load_vectors(vectors_path)  # [N, D], typically unit-normalized
    metas = _load_meta(meta_path)  # list of dicts aligned with rows

    if mat.shape[0] == 0:
        return []

    # Build exact index in memory
    vecs = _ensure_unit_norm(mat).astype(np.float32, copy=False)
    index = _build_flat_index(vecs)

    # Embed query and normalise
    q = _embed_batch([query], backend=backend, model=model, dim=dim)[0]
    if q.shape[0] != vecs.shape[1]:
        raise ValueError(f"query dim {q.shape[0]} != vectors dim {vecs.shape[1]}")
    q = (q / (np.linalg.norm(q) + 1e-12)).astype(np.float32, copy=False)

    # FAISS expects [nq, d], returns (scores, indices)
    scores, idxs = index.search(q.reshape(1, -1), k)
    scores = scores[0]
    idxs = idxs[0]

    out: List[SearchResult] = []
    for rank, (score, i) in enumerate(zip(scores, idxs), start=1):
        # FAISS pads with -1 when not enough vectors
        if i < 0:
            continue

        m = metas[i]
        out.append(
            SearchResult(
                rank=rank,
                score=float(score),
                doc_id=m["doc_id"],
                rel_path=m["rel_path"],
                page=int(m["page"]),
                chunk_id=int(m["chunk_id"]),
            )
        )

    return out
