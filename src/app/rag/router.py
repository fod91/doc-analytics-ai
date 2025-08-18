from fastapi import APIRouter, HTTPException, Query
from app.rag.index_select import search as index_search

router = APIRouter(prefix="/genai", tags=["genai"])


@router.get("", include_in_schema=False)
@router.get("/", summary="GenAI readiness stub")
def genai_ready():
    # Prove router is mounted when FEATURE_GENAI=1
    return {"status": "ready"}


@router.get("/search", summary="Vector search (np only for now)")
def genai_search(
    query: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
    # index vs embedding backends are distinct:
    index_backend: str | None = Query(None, description="'np' or 'faiss'"),
    embed_backend: str | None = Query(None, description="'hash' or 'st'"),
    dim: int | None = Query(
        None, description="embedding dimension (must match vectors)"
    ),
):
    try:
        hits = index_search(
            query,
            k=k,
            index_backend=index_backend,
            embed_backend=embed_backend,
            dim=dim,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "query": query,
        "k": k,
        "index_backend": index_backend,
        "embed_backend": embed_backend,
        "matches": [
            {
                "rank": h.rank,
                "score": h.score,
                "doc_id": h.doc_id,
                "rel_path": h.rel_path,
                "page": h.page,
                "chunk_id": h.chunk_id,
            }
            for h in hits
        ],
    }
