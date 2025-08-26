from __future__ import annotations
from typing import Iterable, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

from app.rag.schema import ChunkRecord
from app.rag.config import get_settings


DEFAULT_CHUNKS = Path("artifacts/chunks/chunks.jsonl")
DEFAULT_EMB_DIR = Path("artifacts/emb")
DEFAULT_VECTORS = DEFAULT_EMB_DIR / "vectors.npy"
DEFAULT_META = DEFAULT_EMB_DIR / "meta.jsonl"
DEFAULT_CHECKSUM = DEFAULT_EMB_DIR / "checksums.txt"


@dataclass(frozen=True)
class Fingerprint:
    chunks_sha256: str
    backend: str
    model: str
    dim: int

    def line(self) -> str:
        return f"{self.chunks_sha256} | backend={self.backend} | model={self.model} | dim={self.dim}"


def _file_sha256(p: Path) -> str:
    """
    Py 3.11+ simple streaming hash.

    FIXME: if we ever need Python <3.11 or custom chunk sizes,
    switch to a manual loop:
        for block in iter(partial(f.read, CHUNK), b""):
            sha.update(block)

    This is to tune chunk size (e.g., 64 KiB / 1 MiB), add progress, or
    update multiple digests in one pass.

    Docs:
      - hashlib.file_digest: https://docs.python.org/3/library/hashlib.html#hashlib.file_digest
      - iter(callable, sentinel) block reader example: https://docs.python.org/3/library/functions.html#iter
      - io.DEFAULT_BUFFER_SIZE reference: https://docs.python.org/3/library/io.html#io.DEFAULT_BUFFER_SIZE
    """
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _fingerprint(chunks_path: Path, backend: str, model: str, dim: int) -> Fingerprint:
    return Fingerprint(
        chunks_sha256=_file_sha256(chunks_path),
        backend=backend,
        model=model,
        dim=dim,
    )


def _iter_chunks(p: Path) -> Iterable[ChunkRecord]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ChunkRecord.model_validate_json(line)


def _hash_vec(texts: List[str], dim: int) -> np.ndarray:
    """Deterministic pseudo-embeddings in [-1,1), then L2-normalize."""
    out = np.empty((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        row = np.empty(dim, dtype=np.float32)
        for d in range(dim):
            h = hashlib.blake2s(f"{t}::{d}".encode(), digest_size=8).digest()
            val = int.from_bytes(h, "big", signed=False)
            row[d] = (val / 2**64) * 2.0 - 1.0
        out[i] = row
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out / norms


def _st_vec(texts: List[str], model_name: str, dim: int) -> np.ndarray:
    # Semantic embeddings via sentence-transformers (CPU)
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except Exception as e:
        raise RuntimeError(
            "Install sentence-transformers and CPU torch to use EMBED_BACKEND=st"
        ) from e

    model = SentenceTransformer(model_name, device="cpu")
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    if vecs.shape[1] != dim:
        rng = np.random.default_rng(0x10C0FFEE)
        proj = rng.normal(size=(vecs.shape[1], dim)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-12
        vecs = vecs @ proj
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs.astype(np.float32)


def _embed_batch(texts: List[str], backend: str, model: str, dim: int) -> np.ndarray:
    if backend == "hash":
        return _hash_vec(texts, dim=dim)
    if backend == "st":
        return _st_vec(texts, model_name=model, dim=dim)
    raise ValueError(f"Unknown backend: {backend}")


def run_embed(
    chunks_path: Path = DEFAULT_CHUNKS,
    out_vectors: Path = DEFAULT_VECTORS,
    out_meta: Path = DEFAULT_META,
    out_checksum: Path = DEFAULT_CHECKSUM,
    *,
    force: bool = False,
    batch_size: int = 64,
    backend: str | None = None,
    model: str | None = None,
    dim: int | None = None,
) -> Tuple[Path, Path]:
    """
    Reads chunk JSONL -> writes vectors.npy (float32) and meta.jsonl (rows aligned).
    Creates/updates checksums.txt with the current fingerprint. If fingerprint matches and
    files exist, re-use cached outputs unless force=True.
    """
    settings = get_settings()
    backend = backend or settings.embed_backend
    model = model or settings.embed_model
    dim = dim or int(settings.embed_dim)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    vectors_dim_mismatch = False
    # out_vectors.parent.mkdir(parents=True, exist_ok=True)
    if out_vectors.exists():
        try:
            _mat = np.load(out_vectors, mmap_mode="r")
            if _mat.ndim != 2 or _mat.shape[1] != dim:
                vectors_dim_mismatch = True
        except Exception:
            # unreadable file? treat as mismatch to force rebuild
            vectors_dim_mismatch = True

    fp = _fingerprint(chunks_path, backend=backend, model=model, dim=dim)
    cached_ok = False
    if (
        out_vectors.exists()
        and out_meta.exists()
        and out_checksum.exists()
        and not vectors_dim_mismatch
        and not force
    ):
        try:
            last_line = (
                out_checksum.read_text(encoding="utf-8").strip().splitlines()[-1]
            )
            if last_line == fp.line():
                cached_ok = True
        except Exception:
            cached_ok = False

    if cached_ok:
        return out_vectors, out_meta

    # Load all chunks into memory (small fixtures), collect texts and meta rows
    texts: List[str] = []
    meta_rows: List[str] = []
    for idx, rec in enumerate(_iter_chunks(chunks_path)):
        texts.append(rec.text)
        meta_rows.append(
            json.dumps(
                {
                    "row": idx,
                    "doc_id": rec.doc_id,
                    "rel_path": rec.rel_path,
                    "page": rec.page,
                    "chunk_id": rec.chunk_id,
                    "span_start": rec.span_start,
                    "span_end": rec.span_end,
                },
                ensure_ascii=False,
            )
        )

    if not texts:
        # write empty files deterministically
        np.save(out_vectors, np.empty((0, dim), dtype=np.float32))
        out_meta.write_text("", encoding="utf-8")
        out_checksum.write_text(fp.line() + "\n", encoding="utf-8")
        return out_vectors, out_meta

    # simple batch-embed loop (hash backend is cheap, but keep the interface)
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # vecs.append(_hash_vec(texts[i : i + batch_size], dim=dim))
        vecs.append(_embed_batch(batch, backend=backend, model=model, dim=dim))
    mat = np.vstack(vecs).astype(np.float32)
    assert mat.shape == (len(texts), dim)

    # Write artifacts
    np.save(out_vectors, mat)
    out_meta.write_text("\n".join(meta_rows) + "\n", encoding="utf-8")
    with out_checksum.open("a", encoding="utf-8") as f:
        f.write(fp.line() + "\n")

    return out_vectors, out_meta


def main():
    # N.B. - uses defaults (dim) for all-MiniLM-L6-v2 model here
    DEFAULT_EMB_DIR.mkdir(parents=True, exist_ok=True)
    vec_p, meta_p = run_embed(
        chunks_path=DEFAULT_CHUNKS,
        out_vectors=DEFAULT_VECTORS,
        out_meta=DEFAULT_META,
        out_checksum=DEFAULT_CHECKSUM,
        backend="hash",
        dim=384,
        batch_size=64,
    )
    print(str(vec_p))
    print(str(meta_p))
    print(str(DEFAULT_CHECKSUM))


if __name__ == "__main__":
    main()
