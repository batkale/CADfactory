import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

logger = logging.getLogger("cadfactory.database")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cadfactory.db")
logger.info(f"Database URL: {DATABASE_URL}")

# connect_args only needed for SQLite
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

try:
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
    # Test the connection immediately so startup fails fast with a clear error
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("OK: Database connection ready")
except Exception as e:
    logger.error(f"FAIL: Cannot connect to database ({DATABASE_URL}): {e}")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"DB session error: {e}", exc_info=True)
        raise
    finally:
        db.close()
