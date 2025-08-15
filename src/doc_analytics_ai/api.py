from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
import threading
import time
import logging

from .db import Base, engine

# models import - needed to establish tables if DB not up on startup
# otherwise psql will not see any relations
from . import models  # noqa: F401

log = logging.getLogger("doc-analytics-ai")

app = FastAPI(title="doc-analytics-ai", version="0.1.0")

DB_READY = False


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


@app.on_event("startup")
def on_startup():
    # kick off a background thread; don't block app startup
    t = threading.Thread(target=_background_db_ensurer, daemon=True)
    t.start()


@app.get("/health")
def health():
    return {"status": "ok", "db_ready": DB_READY}
