from __future__ import annotations
from typing import List, Optional

from app.rag.config import get_settings
from app.rag.index_np import search_np, SearchResult as NPResult


SearchResult = NPResult


def search(
    query: str,
    k: int = 5,
    *,
    # which vector index to use
    index_backend: Optional[str] = None,
    # which embedding to use for the query vector
    embed_backend: Optional[str] = None,
    model: Optional[str] = None,
    dim: Optional[int] = None,
    **paths,
) -> List[SearchResult]:
    """
    Dispatch to NumPy (or FAISS when installed later) search.
    - index_backend: selects the ANN/linear index ('np' default; 'faiss' if installed).
    - embed_backend: selects how the query is embedded ('hash' default; 'st' if installed).
    """
    s = get_settings()
    which_index = (index_backend or s.index_backend or "np").lower()
    which_embed = (embed_backend or s.embed_backend or "hash").lower()
    model = model or s.embed_model
    dim = int(dim or s.embed_dim)

    if which_index == "np":
        return search_np(
            query,
            k=k,
            backend=which_embed,
            model=model,
            dim=dim,
            **paths,
        )

    raise ValueError(f"Unknown index backend: {which_index!r}")
