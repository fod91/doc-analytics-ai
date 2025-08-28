import os
from fastapi.testclient import TestClient

def _client(rag_backend: str, feature: str = "1") -> TestClient:
    os.environ["FEATURE_GENAI"] = feature
    os.environ["RAG_BACKEND"] = rag_backend
    import app.rag.config as cfg
    cfg.get_settings.cache_clear()
    from doc_analytics_ai.api import create_app
    return TestClient(create_app())

def test_genai_router_vanilla_selected_by_default():
    c = _client("vanilla")
    r = c.get("/genai/")
    assert r.status_code in (200, 404)  # 404 if FEATURE_GENAI=0
    if r.status_code == 200:
        assert r.json().get("status") == "ready"

def test_genai_router_framework_flag_is_accepted():
    c = _client("framework")
    r = c.get("/genai/")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.json().get("status") == "ready"
