from __future__ import annotations
from pathlib import Path
from typing import Iterator, Tuple, Optional

from app.rag.config import get_settings


def _iter_pypdf(path: Path, *, max_pages: Optional[int]) -> Iterator[Tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise ImportError("pypdf not installed") from e

    r = PdfReader(str(path))
    for i, page in enumerate(r.pages, start=1):
        if max_pages and i > max_pages:
            break
        txt = page.extract_text() or ""
        yield i, txt


def iter_pdf_pages(
    path: Path,
    *,
    backend: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> Iterator[Tuple[int, str]]:
    # Yield (page_number, text) for each page in the PDF at `path`
    b = (backend or get_settings().pdf_backend or "pypdf").lower()
    if b == "pypdf":
        yield from _iter_pypdf(path, max_pages=max_pages)
        return

    # Future backends plug in here (pdfminer/pdfplumber/fitz)
    # For now: do nothing if unsupported backend selected (graceful no-op)
    return
