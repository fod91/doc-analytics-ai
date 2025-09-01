from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path

from app.genai.ingest import load_dir
# Prefer the standalone package if present
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 120

def normalise_newlines(text: str) -> str:
    # Make splitting deterministic across platforms
    return text.replace("\r\n", "\n").replace("\r", "\n")

def sort_key(d: Dict[str, Any]) -> Tuple[str, int]:
    m = d.get("metadata", {})
    return (str(m.get("src_path", "")), int(m.get("page", 1)))

def make_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Deterministic, conservative separators to keep context boundaries sensible
        separators=["\n\n", "\n", " ", ""],
        # Keep separators so overlaps are literal (helps exact boundary checks in tests)
        keep_separator=False,  # False = cleaner chunks; determinism still holds
    )

def split_docs(
    docs: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    # Sorting: by (src_path, page) for determinism
    splitter = make_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Dict[str, Any]] = []
    for d in sorted(docs, key=sort_key):
        text = normalise_newlines(d.get("page_content", "") or "")
        md = dict(d.get("metadata") or {})
        doc_id = str(md.get("doc_id", ""))
        page = int(md.get("page", 1))
        # Split this page's text
        parts = splitter.split_text(text)
        for i, part in enumerate(parts):
            cmeta = dict(md)
            cmeta["chunk_index"] = i
            cmeta["chunk_id"] = f"{doc_id}:{page}:{i:04d}"
            chunks.append({"page_content": part, "metadata": cmeta})
    return chunks

def split_dir(
    root: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    return split_docs(load_dir(root), chunk_size=chunk_size, chunk_overlap=chunk_overlap)

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("usage: python -m app.genai.splitter <SRC_DIR> [CHUNK_SIZE] [CHUNK_OVERLAP]")
        sys.exit(2)
    src = sys.argv[1]
    cs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CHUNK_SIZE
    co = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CHUNK_OVERLAP
    out = split_dir(src, chunk_size=cs, chunk_overlap=co)
    # Light summary
    by_src = {}
    for c in out:
        s = c["metadata"]["src_path"]
        by_src[s] = by_src.get(s, 0) + 1
    print(json.dumps({
        "source": src,
        "total_chunks": len(out),
        "by_src": by_src,
        "sample": out[:2],
    }, ensure_ascii=False))
