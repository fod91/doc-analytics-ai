from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Iterable, List, Dict, Any

# Note - lazy imports so the file does not pull heavy dependencies unless used
def load_pdf(path: Path):
    # PyPDFLoader from langchain_community; keeps deps minimal (pypdf)
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(path))
    return loader.load()

def load_text(path: Path):
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    return loader.load()

def doc_id(src_path: str) -> str:
    return hashlib.sha1(src_path.encode("utf-8")).hexdigest()[:16]

def normalise_docs(docs: Iterable[Any], src_path: str) -> List[Dict[str, Any]]:
    # Return list of plain dictionaries
    out: List[Dict[str, Any]] = []
    base_meta = {"doc_id": doc_id(src_path), "src_path": src_path}
    for d in docs:
        meta = dict(getattr(d, "metadata", {}) or {})
        # TextLoader has no page - PDF pages set by PyPDFLoader
        page = meta.get("page", 1)
        merged = {**base_meta, **meta, "page": page}
        out.append({"page_content": getattr(d, "page_content", ""), "metadata": merged})
    return out

def load_path(path: str | Path) -> List[Dict[str, Any]]:
    # Load a single file (pdf/txt/md) into normalised documents
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return normalise_docs(load_pdf(p), str(p))
    if suffix in (".txt", ".md"):
        return normalise_docs(load_text(p), str(p))
    # Unsupported files are skipped gracefully
    return []

def iter_files(root: str | Path, patterns: tuple[str, ...] = ("*.pdf", "*.txt", "*.md")) -> Iterable[Path]:
    root = Path(root)
    for pat in patterns:
        yield from root.rglob(pat)

def load_dir(root: str | Path) -> List[Dict[str, Any]]:
    # Load supported files under root into normalised document dictionaries
    all_docs: List[Dict[str, Any]] = []
    for f in iter_files(root):
        all_docs.extend(load_path(f))
    return all_docs

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("usage: python -m app.genai.adapters.ingest <SRC_DIR>")
        sys.exit(2)
    src = sys.argv[1]
    docs = load_dir(src)
    print(json.dumps({
        "source": src,
        "files": len(list(iter_files(src))),
        "docs": len(docs),
        "sample": docs[:2],
    }, ensure_ascii=False))
