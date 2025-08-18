from pathlib import Path
import hashlib
from app.rag.embed import _file_sha256


def test_file_sha256_matches_python_reference(tmp_path: Path):
    data = b"abc" * 1000 + b"\x00\x01\x02"
    f = tmp_path / "blob.bin"
    f.write_bytes(data)

    # reference: in-memory hash
    ref = hashlib.sha256(data).hexdigest()
    # function under test
    got = _file_sha256(f)
    assert got == ref and len(got) == 64
