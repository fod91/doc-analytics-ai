import os
import requests
import streamlit as st

API_BASE = os.getenv("DOC_AI_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Document Analytics AI", layout="centered")
st.title("Document Analytics AI — Sprint 1")

# Health
st.subheader("Health")
try:
    h = requests.get(f"{API_BASE}/health", timeout=2).json()
    st.json(h)
except Exception as e:
    st.error(f"Health check failed: {e}")

# Upload
st.subheader("Upload to Object Storage")
f = st.file_uploader("Choose a file")
source = st.text_input("Source (optional)")
if st.button("Upload", disabled=(f is None)):
    files = {"file": (f.name, f.getvalue(), "type" if hasattr(f, "type") else None)}
    data = {"source": source} if source else {}
    r = requests.post(f"{API_BASE}/upload", files=files, data=data)
    if r.ok:
        st.success("Uploaded")
        st.json(r.json())
    else:
        st.error(r.text)

# Ingest
st.subheader("Ingest texts")
ingest_json = st.text_area(
    "Paste JSON array of {source, text, label?}",
    height=150,
    value='[{"source":"ui","text":"Great!","label":"pos"}]',
)
if st.button("Ingest batch"):
    try:
        # quick-and-dirty for now, replace with json.loads later
        r = requests.post(f"{API_BASE}/ingest", json=eval(ingest_json))
        st.json(r.json())
    except Exception as e:
        st.error(str(e))

# Analytics
st.subheader("Sentiment summary")
if st.button("Refresh summary"):
    r = requests.get(f"{API_BASE}/analytics/sentiment")
    if r.ok:
        st.json(r.json())
    else:
        st.error(r.text)
