from pathlib import Path
import hashlib


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
