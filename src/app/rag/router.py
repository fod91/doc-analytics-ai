from fastapi import APIRouter

router = APIRouter(prefix="/genai", tags=["genai"])


@router.get("", include_in_schema=False)
@router.get("/", summary="GenAI readiness stub")
def genai_ready():
    # Prove router is mounted when FEATURE_GENAI=1
    return {"status": "ready"}
