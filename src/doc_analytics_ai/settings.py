import os
from dotenv import load_dotenv

load_dotenv()

_AWS_DEFAULT_REGION = "eu-west-1"
REQUIRED_VARS = ["PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"]


def _getenv_required(name: str) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        raise RuntimeError(
            f"Missing required environment variable: {name}. Create a .env from and set all necessary variables"
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


# MinIO / local defaults
S3_ENDPOINT = _getenv_required("MINIO_ENDPOINT")
S3_BUCKET = _getenv_required("MINIO_BUCKET")

# Credentials (MinIO or AWS)
S3_ACCESS_KEY = os.getenv("MINIO_ROOT_USER") or os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("AWS_SECRET_ACCESS_KEY")

if not S3_ACCESS_KEY or not S3_SECRET_KEY:
    raise RuntimeError(
        "Missing S3 credentials. Provide MINIO_ROOT_USER/MINIO_ROOT_PASSWORD or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY."
    )

SQLALCHEMY_URL: str = get_sqlalchemy_url()
# Region (AWS requires; MinIO tolerates any)
S3_REGION = os.getenv("AWS_DEFAULT_REGION", _AWS_DEFAULT_REGION)
