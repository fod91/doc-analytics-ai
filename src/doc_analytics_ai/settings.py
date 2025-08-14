import os

# from typing import Optional

from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = ["PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"]


def _getenv_required(name: str) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        raise RuntimeError(
            f"Missing required environment variable: {name}. Create a .env from and set all Postgres values, or provide PG_DSN."
        )
    return v


def _pg_dsn_from_parts() -> str:
    host = _getenv_required("PGHOST")
    port = _getenv_required("PGPORT")
    user = _getenv_required("PGUSER")
    pwd = _getenv_required("PGPASSWORD")
    db = _getenv_required("PGDATABASE")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def get_sqlalchemy_url() -> str:
    # Prefer a single DSN if provided; otherwise build from required parts
    dsn = os.getenv("PG_DSN")
    return dsn if dsn else _pg_dsn_from_parts()


SQLALCHEMY_URL: str = get_sqlalchemy_url()
