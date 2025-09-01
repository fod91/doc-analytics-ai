from app.genai.splitter import split_docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def make_doc(text: str, src: str, page: int, doc_id: str):
    return {"page_content": text, "metadata": {"src_path": src, "page": page, "doc_id": doc_id}}

def test_chunk_sizes_and_overlap_simple():
    # Use uniform text (no separators) so overlap behavior is exact
    text = "x" * 2600
    docs = [make_doc(text, "a.txt", 1, "docA")]
    # out = split_docs(docs, chunk_size=1000, chunk_overlap=120)
    out = split_docs(docs, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    # Assuming DEFAULT_CHUNK_SIZE = 1000 and DEFAULT_CHUNK_OVERLAP = 120
    # Expect 3 chunks: 1000, 1000, 840 (stride = 1000-120 = 880; starts at 0,880,1760)
    lens = list(map(lambda d: len(d["page_content"]), out))
    assert lens[0] == DEFAULT_CHUNK_SIZE
    assert lens[1] == DEFAULT_CHUNK_SIZE
    assert lens[2] == 840
    # Overlap checks: suffix of previous == prefix of next (120 chars)
    assert out[0]["page_content"][-DEFAULT_CHUNK_OVERLAP:] == out[1]["page_content"][:DEFAULT_CHUNK_OVERLAP]
    assert out[1]["page_content"][-DEFAULT_CHUNK_OVERLAP:] == out[2]["page_content"][:DEFAULT_CHUNK_OVERLAP]
    # Deterministic IDs
    assert out[0]["metadata"]["chunk_id"].endswith(":0000")
    assert out[1]["metadata"]["chunk_id"].endswith(":0001")
    assert out[2]["metadata"]["chunk_id"].endswith(":0002")

def test_ordering_by_src_then_page():
    d1 = make_doc("a"*500, "b.md", 2, "B")
    d2 = make_doc("b"*500, "a.md", 1, "A")
    out = split_docs([d1, d2], chunk_size=200, chunk_overlap=50)
    # first chunk should come from a.md page 1 due to sort order
    assert out[0]["metadata"]["src_path"] == "a.md"
    assert out[0]["metadata"]["page"] == 1

def test_defaults_are_stable():
    d = make_doc("z"*2000, "p.pdf", 3, "D")
    out1 = split_docs([d])
    out2 = split_docs([d])
    assert [c["metadata"]["chunk_id"] for c in out1] == [c["metadata"]["chunk_id"] for c in out2]
