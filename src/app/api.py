from fastapi import FastAPI
from app.rag.config import get_settings
from app.rag.router import router as genai_router


def create_app() -> FastAPI:
    app = FastAPI(title="Doc Analytics AI", version="0.0.1")

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    settings = get_settings()
    if settings.feature_genai:
        app.include_router(genai_router)
    return app


app = create_app()
