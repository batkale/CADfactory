import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
File storage service — saves uploaded CAD files to local disk.
Easily swappable for S3 / GCS in production.
"""

import os  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import aiofiles  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024
ALLOWED_EXTENSIONS = {".stl", ".3mf", ".stp", ".step"}


def _ensure_upload_dir():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def generate_stored_filename(original_filename: str) -> str:
    ext = Path(original_filename).suffix.lower()
    return f"{uuid.uuid4().hex}{ext}"


async def save_file(data: bytes, stored_filename: str) -> str:
    _ensure_upload_dir()
    path = UPLOAD_DIR / stored_filename
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)
    return str(path)


async def read_file(stored_filename: str) -> bytes:
    path = UPLOAD_DIR / stored_filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {stored_filename}")
    async with aiofiles.open(path, "rb") as f:
        return await f.read()


def delete_file(stored_filename: str) -> bool:
    path = UPLOAD_DIR / stored_filename
    if path.exists():
        path.unlink()
        return True
    return False


def validate_file(filename: str, size_bytes: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large ({size_bytes // 1024 // 1024} MB). "
            f"Max: {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB"
        )
