import os
from fastapi.testclient import TestClient


def build_client_with_env(flag: str) -> TestClient:
    os.environ["FEATURE_GENAI"] = flag
    # Clear cached settings so env changes take effect
    import app.rag.config as cfg

    cfg.get_settings.cache_clear()
    # Build a fresh app instance so gating re-evaluates
    from app.api import create_app

    return TestClient(create_app())


def test_genai_router_disabled_by_default():
    client = build_client_with_env("0")
    r = client.get("/genai")
    assert r.status_code == 404


def test_genai_router_enabled_when_flag_set():
    client = build_client_with_env("1")
    r = client.get("/genai")
    assert r.status_code == 200
    assert r.json().get("status") == "ready"
