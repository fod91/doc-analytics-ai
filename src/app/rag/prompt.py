from __future__ import annotations
from typing import List, Dict


def build_prompt(query: str, contexts: List[Dict]) -> str:
    # Basic prompt: include query and short snippets
    lines = [f"QUESTION: {query}", "", "CONTEXTS:"]
    for c in contexts:
        snippet = (c.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "…"
        lines.append(f"- [{c['rank']}] {c['rel_path']}#{c['chunk_id']} :: {snippet}")
    lines.append("")
    lines.append(
        "INSTRUCTIONS: Answer using the contexts. If uncertain, say so. Provide brief citations like [rank]."
    )
    return "\n".join(lines)
