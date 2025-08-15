from fastapi import FastAPI
from .db import Base, engine

app = FastAPI(title="doc-analytics-ai", version="0.1.0")

# create tables on startup
Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok"}
