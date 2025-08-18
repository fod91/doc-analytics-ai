from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple


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
    """
    Yield (relative_path, text) for .md and .txt files under src_root
    Ignores PDFs for now...
    """
    if not src_root.exists():
        return

    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
            rel = str(p.relative_to(src_root))
            yield rel, _read_text_file(p)


def records_from_text(rel_path: str, text: str) -> Dict[str, Any]:
    # Create a single JSON record for the whole file (page=1). Chunking comes later
    return {
        "doc_id": _stable_doc_id(rel_path),
        "rel_path": rel_path,
        "page": 1,
        "span_start": 0,
        "span_end": len(text),
        "text": text,
    }


def run_ingest(src_root: Path = DEFAULT_SRC, out_file: Path = DEFAULT_OUT_FILE) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_file.open("w", encoding="utf-8") as f:
        for rel, text in iter_text_docs(src_root):
            rec = records_from_text(rel, text)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        # Still succeed, but leave an empty file present for deterministic pipelines
        out_file.touch(exist_ok=True)
    return out_file


def main():
    out = run_ingest(DEFAULT_SRC, DEFAULT_OUT_FILE)
    print(str(out))


if __name__ == "__main__":
    main()
