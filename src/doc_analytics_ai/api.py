from fastapi import FastAPI, Depends, UploadFile, File
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
import threading
import time
import logging
import uuid

from .db import Base, engine, SessionLocal
from .schema import IngestItem
from .s3 import ensure_bucket_if_missing, upload_fileobj, presigned_get_url

# models import - needed to establish tables if DB not up on startup
# otherwise psql will not see any relations
from . import models  # noqa: F401

log = logging.getLogger("doc-analytics-ai")

app = FastAPI(title="doc-analytics-ai", version="0.1.0")

DB_READY = False
S3_READY = False


def _probe_and_init() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT current_user, current_database()"))
        Base.metadata.create_all(bind=engine)
        return True
    except OperationalError:
        return False


def _background_db_ensurer(interval: float = 2.0, max_seconds: int = 300):
    global DB_READY
    deadline = time.time() + max_seconds
    while time.time() < deadline and not DB_READY:
        if _probe_and_init():
            DB_READY = True
            log.info("Database is ready and tables are ensured.")
            return
        time.sleep(interval)
    if not DB_READY:
        log.error("Database did not become ready within %ss.", max_seconds)


def _background_s3_ensurer(interval: float = 2.0, max_seconds: int = 120) -> None:
    global S3_READY
    deadline = time.time() + max_seconds
    while time.time() < deadline and not S3_READY:
        try:
            ensure_bucket_if_missing()
            S3_READY = True
            log.info("S3 bucket is ready.")
            return
        except Exception as e:
            log.warning("S3 not ready yet: %s", e)
            time.sleep(interval)
    if not S3_READY:
        log.error("S3 did not become ready within %ss.", max_seconds)


@app.on_event("startup")
def on_startup():
    # kick off a background thread; don't block app startup
    threading.Thread(target=_background_db_ensurer, daemon=True).start()
    threading.Thread(target=_background_s3_ensurer, daemon=True).start()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok", "db_ready": DB_READY, "s3_ready": S3_READY}


@app.post("/ingest")
def ingest(items: list[IngestItem], db: Session = Depends(get_db)):
    # Persist a batch of rows (source, text, optional label)
    objs = [
        models.Transcript(source=i.source, text=i.text, label=i.label) for i in items
    ]
    db.add_all(objs)
    db.commit()
    return {"ingested": len(objs)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Store a single file in S3/MinIO using boto3
    # Returns object key and a presigned GET URL
    if not S3_READY:
        return {"ok": False, "error": "storage not ready"}

    # Generate a key; keep file extension if present
    suffix = ""
    if "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    key = f"uploads/{uuid.uuid4().hex}{suffix}"

    # Stream upload without loading into memory
    upload_fileobj(file.file, key, content_type=file.content_type)
    url = presigned_get_url(key, expires_seconds=1800)
    return {"ok": True, "key": key, "url": url}
