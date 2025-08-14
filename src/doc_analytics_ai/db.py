from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .settings import SQLALCHEMY_URL

# - pool_pre_ping=True: validates connections before use; avoids "server closed"
#   errors after idle time by discarding dead connections.
# - future=True: opt into SQLAlchemy 2.0-style API
engine = create_engine(
    SQLALCHEMY_URL,
    pool_pre_ping=True,
    future=True,
)

# Session:
# - autocommit=False: explicit transaction boundaries; you must call commit()
# - autoflush=False: avoid implicit flushes; improves predictability for tests
# - future=True: opt into SQLAlchemy 2.0-style API
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)

Base = declarative_base()
