from __future__ import annotations
from typing import List, Dict
from app.rag.config import get_settings


def generate_answer(
    prompt: str, contexts: List[Dict], *, backend: str | None = None
) -> str:
    # Deterministic 'mock' answer - echoes first 1-2 context snippets with [rank] citations
    s = get_settings()
    b = (backend or ("mock" if s.mock_llm else "mock")).lower()
    if b != "mock":
        # reserved for future real LLM integration
        raise RuntimeError("Only a mock LLM backend is available in this commit")

    # Build a tiny extractive answer
    parts = []
    for c in contexts[:2]:
        snippet = (c.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200].rstrip() + "..."
        parts.append(f"{snippet} [{c['rank']}]")
    if not parts:
        return "I do not know."
    return " ".join(parts)
