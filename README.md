# doc-analytics-ai

FastAPI + Postgres + S3/MinIO service for document ingestion, storage, and basic analytics — extended with a sprint-2 **RAG** pipeline (chunk → embed → index → retrieve) and a mock LLM for `/genai/ask`.

> The `/genai/*` endpoints are **feature-gated**. Set `FEATURE_GENAI=1` to enable.

---

## Features
- **Core API**
  - `POST /ingest` — store labeled text rows in Postgres.
  - `POST /upload` — upload binary files to S3/MinIO and persist metadata.
  - `GET /analytics/sentiment` — label counts across ingested rows.
  - `GET /health` — DB/S3 readiness.
- **RAG (feature-gated)**
  - Deterministic chunking, embeddings (hash or SentenceTransformers), NumPy/FAISS indexing, keyword re-rank, mock LLM answer composition.
  - `GET /genai/search` and `GET /genai/ask` with citations.
- **Dev UX**
  - Streamlit mini UI to poke API.
  - Tests for core analytics + RAG.

---

## Requirements
- Python **3.11**
- Postgres **16** (Docker Compose provided)
- MinIO (Docker Compose provided) or AWS S3
- Optional for RAG quality/scale:
  - `sentence-transformers` (and CPU `torch`) for `EMBED_BACKEND=st`
  - `faiss-cpu` for `INDEX_BACKEND=faiss`
  - `pypdf>=6` for PDF extraction (helper scaffolded)

---

## Setup
### 1) Create environment
**Conda (recommended)**
```bash
conda env create -f environment.yml
conda activate doc_analytics_ai
```
**or venv & pip**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment variables
Copy `.env.example` to `.env` and adjust as needed:
```env
PGHOST=localhost
PGPORT=5432
PGUSER=admin
PGPASSWORD=docai
PGDATABASE=docai
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_ENDPOINT=http://localhost:9000
MINIO_BUCKET=docai-dev
# RAG (feature-gated)
FEATURE_GENAI=0
EMBED_BACKEND=hash     # or st
EMBED_DIM=384          # must match vectors
INDEX_BACKEND=faiss    # or np
RERANK_STRATEGY=auto   # or keyword, none
CANDIDATE_MULTIPLIER=4
```

### 3) Start infrastructure (Postgres + MinIO)
```bash
docker compose up -d postgres minio
```
> Note: `docker-compose.yml` also defines an `api` service with `build: .` but no Dockerfile yet (see **Known gaps**). Run the API directly for now.

### 4) Run the API
```bash
# ensure imports resolve in editable dev mode
export PYTHONPATH=src
# launch uvicorn via module entrypoint
python -m doc_analytics_ai.main
# use this if you started app with '&': pkill -f "python -m doc_analytics_ai.main"
```
Alternative:
```bash
uvicorn doc_analytics_ai.api:app --reload
```

### 5) Streamlit demo (optional)
```bash
export DOC_AI_API_BASE=http://localhost:8000
streamlit run apps/ui_streamlit.py
```

---

## Endpoints
### Health
```bash
curl -s http://localhost:8000/health | jq
```

### Ingest (JSONL-style array of rows)
```bash
curl -s -X POST http://localhost:8000/ingest \
  -H 'content-type: application/json' \
  -d '[{"source":"demo","text":"great!","label":"pos"},{"source":"demo","text":"bad","label":"neg"}]'
```

### Upload to MinIO/S3
```bash
curl -s -F 'file=@README.md' -F 'source=demo' http://localhost:8000/upload | jq
```

### Sentiment summary
```bash
curl -s http://localhost:8000/analytics/sentiment | jq
```

---

## RAG quickstart
Enable feature and build a tiny demo corpus from fixtures.
```bash
export FEATURE_GENAI=1
python - <<'PY'
from app.rag.ingest import run_ingest, DEFAULT_SRC
from app.rag.chunk import run_chunker
from app.rag.embed import run_embed
run_ingest(DEFAULT_SRC)            # -> artifacts/ingest/docs.jsonl
run_chunker()                      # -> artifacts/chunks/chunks.jsonl
run_embed(dim=32, backend='hash')  # -> artifacts/emb/{vectors.npy,meta.jsonl}
PY
```
Artifacts are written under `artifacts/`:
```
ingest/docs.jsonl
chunks/chunks.jsonl
emb/vectors.npy
emb/meta.jsonl
emb/checksums.txt
```

### `/genai` endpoints (enabled when `FEATURE_GENAI=1`)
- `GET /genai/` → `{status:"ready"}`
- `GET /genai/search` — params: `query`, `k` (1..50), `index_backend` (`np|faiss`), `embed_backend` (`hash|st`), `dim`
- `GET /genai/ask` — params: `query`, `k` (1..10), `index_backend`, `embed_backend`, `dim`, `rerank` (`none|auto|keyword`), `candidate_multiplier`

#### Example cURL
**Keyword re-rank (per-call override):**
```bash
curl -Gs --data-urlencode 'query=barrow-blade' \
  --data-urlencode 'rerank=keyword' \
  --data-urlencode 'candidate_multiplier=5' \
  'http://127.0.0.1:8000/genai/ask' | jq
```
**SentenceTransformers vectors (384-dim):**
```bash
# Requires: pip install sentence-transformers torch
curl -Gs --data-urlencode 'query=Quickbeam' \
  --data-urlencode 'embed_backend=st' \
  --data-urlencode 'dim=384' \
  --data-urlencode 'rerank=auto' \
  'http://127.0.0.1:8000/genai/ask' | jq
```

---

## Development workflow
```bash
# once the env is created
conda activate doc_analytics_ai
export PYTHONPATH=src

# run app (foreground)
python -m doc_analytics_ai.main

# lint & tests
conda run -n doc_analytics_ai ruff check src tests
conda run -n doc_analytics_ai pytest -q --cov=app --cov=doc_analytics_ai --cov-report=term-missing
```

---

## Configuration reference
**Postgres**: `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`

**S3/MinIO**:
- `MINIO_ENDPOINT` (e.g., `http://localhost:9000`)
- `MINIO_BUCKET` (e.g., `docai-dev`)
- Credentials: `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` **or** `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`
- Optional: `AWS_DEFAULT_REGION` (default `eu-west-1`)

**RAG** (pydantic-settings; case-insensitive):
- `FEATURE_GENAI` (default `0`)
- `EMBED_BACKEND` (`hash|st`, default `hash`)
- `EMBED_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`)
- `EMBED_DIM` (default `384`)
- `INDEX_BACKEND` (`faiss|np`, default `faiss` when installed, otherwise `np`)
- `RERANK_STRATEGY` (`auto|keyword|none`, default `auto`)
- `CANDIDATE_MULTIPLIER` (default `4`)
- `PDF_BACKEND` (default `pypdf`)

---

## Known gaps / TODO
- **Dockerfile** missing (compose references `api` service). Either add a Dockerfile or remove the service stanza.
- **Package layout** mixes `doc_analytics_ai` and `app.rag`. For a cleaner import story, consider moving `app.rag` under `doc_analytics_ai.rag` (or add an import alias) and extend `pyproject.toml` to include both packages.
- **Dependencies** exist in both `requirements.txt` and `pyproject.toml`. Pick one source of truth (e.g., keep `requirements.txt` for runtime + `requirements-dev.txt` for tooling) or switch fully to PEP 621 with extras.
- **Tests coverage** currently targets `app`; include `--cov=doc_analytics_ai` (see command above).
- **Streamlit** uses `eval()` for JSON in the ingest panel; replace with `json.loads()` for safety.

---

## License
MIT (if applicable).

