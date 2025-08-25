from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple, Iterator

from app.rag.config import get_settings
from app.rag.pdf_extract import iter_pdf_pages

log = logging.getLogger("doc-analytics-ai")


DEFAULT_SRC = Path("evals/fixtures")
DEFAULT_OUT_DIR = Path("artifacts/ingest")
DEFAULT_OUT_FILE = DEFAULT_OUT_DIR / "docs.jsonl"


def _stable_doc_id(rel_path: str) -> str:
    # Stable identifier derived from the relative path (no content dependency)
    h = hashlib.blake2s(rel_path.encode("utf-8"), digest_size=16).hexdigest()
    stem = Path(rel_path).stem
    return f"{stem}-{h[:8]}"


def _read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def iter_text_docs(src_root: Path) -> Iterable[Tuple[str, str]]:
    # Yield (relative_path, text) for .md and .txt files under src_root
    if not src_root.exists():
        return

    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
            rel = str(p.relative_to(src_root))
            yield rel, _read_text_file(p)


def records_from_text(rel_path: str, text: str, *, page: int = 1) -> Dict[str, Any]:
    # Create a single JSON record for the whole file (page=1 for text files, real page number for PDFs). Chunking comes later
    return {
        "doc_id": _stable_doc_id(rel_path),
        "rel_path": rel_path,
        "page": page,
        "span_start": 0,
        "span_end": len(text),
        "text": text,
    }


def iter_records_from_source(source: Path) -> Iterator[Dict[str, Any]]:
    """
    Unified generator:
      - If `source` is a directory: walk .md/.txt and .pdf (per page)
      - If `source` is a file: emit just that file (per page for PDFs)
    """
    if source.is_dir():
        # Text files (whole file)
        for rel, text in iter_text_docs(source):
            yield records_from_text(rel, text, page=1)
        # PDFs (per page)
        for rel, page_no, text in iter_pdf_docs(source):
            yield records_from_text(rel, text, page=page_no)
        return

    if source.is_file():
        ext = source.suffix.lower()
        # Use the filename as rel_path when ingesting a single file
        rel = source.name
        if ext in {".md", ".txt"}:
            yield records_from_text(rel, _read_text_file(source), page=1)
            return
        if ext == ".pdf":
            backend = get_settings().pdf_backend
            try:
                for page_no, text in iter_pdf_pages(
                    source, backend=backend, max_pages=None
                ):
                    yield records_from_text(rel, text, page=page_no)
            except Exception as e:
                log.warning("PDF extraction failed for %s via %s: %s", rel, backend, e)
            return
        # Unsupported file: no-op
        log.info("Skipping unsupported file type: %s", source)
        return

    log.warning("Source not found: %s", source)


def iter_pdf_docs(src_root: Path) -> Iterable[Tuple[str, int, str]]:
    # Yield (rel_path, page_number, text) for every PDF page under src_root,
    # using the backend defined in settings.PDF_BACKEND (default: 'pypdf')
    if not src_root.exists():
        return

    backend = get_settings().pdf_backend
    for pdf in src_root.rglob("*.pdf"):
        if not pdf.is_file():
            continue
        rel = str(pdf.relative_to(src_root))
        try:
            for page_no, text in iter_pdf_pages(pdf, backend=backend, max_pages=None):
                yield rel, page_no, text
        except Exception as e:
            log.warning("PDF extraction failed for %s for %s: %s", rel, backend, e)


def run_ingest(src_root: Path = DEFAULT_SRC, out_file: Path = DEFAULT_OUT_FILE) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_file.open("w", encoding="utf-8") as f:
        for rec in iter_records_from_source(src_root):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        # Keep an empty file for deterministic pipelines
        out_file.touch(exist_ok=True)
    log.info("Ingest finished: %d records (source=%s) -> %s", count, src_root, out_file)
    return out_file


def main():
    out = run_ingest(DEFAULT_SRC, DEFAULT_OUT_FILE)
    print(str(out))


if __name__ == "__main__":
    main()
