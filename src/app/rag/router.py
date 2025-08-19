from fastapi import APIRouter, HTTPException, Query, Depends
from app.rag.index_select import search as index_search
from app.rag.config import get_settings
from app.rag import retrieve as retriever
from app.rag import prompt as prompt_builder
from app.rag import llm as llm_mod


def _ensure_enabled():
    # If feature is off, pretend the routes aren't there
    if not get_settings().feature_genai:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(
    prefix="/genai",
    tags=["genai"],
    dependencies=[Depends(_ensure_enabled)],
)


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


@router.get("/ask", summary="Answer a question using retrieved context (mock LLM)")
def genai_ask(
    query: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=10),
    index_backend: str | None = Query(
        None, description="'np'"
    ),  # or faiss when available
    embed_backend: str | None = Query(None, description="'hash' or 'st'"),
    dim: int | None = Query(
        None, description="embedding dimension (must match vectors)"
    ),
):
    try:
        contexts = retriever.retrieve(
            query,
            k=k,
            index_backend=index_backend,
            embed_backend=embed_backend,
            dim=dim,
        )
        prompt = prompt_builder.build_prompt(query, contexts)
        answer = llm_mod.generate_answer(prompt, contexts, backend="mock")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "query": query,
        "k": k,
        "answer": answer,
        "citations": [
            {
                "rank": c["rank"],
                "score": c["score"],
                "doc_id": c["doc_id"],
                "rel_path": c["rel_path"],
                "page": c["page"],
                "chunk_id": c["chunk_id"],
            }
            for c in contexts
        ],
    }
