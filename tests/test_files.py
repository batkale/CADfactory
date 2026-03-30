"""
Integration tests for /files endpoints.
"""
import io
import os
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_files.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-ci-only")
os.environ.setdefault("UPLOAD_DIR", "/tmp/cadfactory_test_uploads")

from database import Base, get_db  # noqa: E402
from main import app  # noqa: E402

TEST_DB_URL = "sqlite:///./test_files.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    os.makedirs("/tmp/cadfactory_test_uploads", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app, raise_server_exceptions=False)

USER = {"email": "files@example.com", "username": "filesuser", "password": "StrongPass1!"}


def _get_token() -> str:
    resp = client.post("/auth/register", json=USER)
    if resp.status_code == 409:
        resp = client.post("/auth/login", data={"username": USER["email"], "password": USER["password"]})
    return resp.json()["access_token"]


# Minimal valid binary STL (80 byte header + 4-byte triangle count = 0)
MINIMAL_STL = b"\x00" * 80 + b"\x00\x00\x00\x00"


def test_upload_stl():
    token = _get_token()
    resp = client.post(
        "/files/upload",
        files={"file": ("part.stl", io.BytesIO(MINIMAL_STL), "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "part.stl"
    assert data["file_format"] == "STL"
    return data["id"]


def test_upload_invalid_extension():
    token = _get_token()
    resp = client.post(
        "/files/upload",
        files={"file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


def test_list_files_empty():
    token = _get_token()
    resp = client.get("/files/", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_files_after_upload():
    token = _get_token()
    client.post(
        "/files/upload",
        files={"file": ("part.stl", io.BytesIO(MINIMAL_STL), "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )
    resp = client.get("/files/", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_file_by_id():
    token = _get_token()
    up = client.post(
        "/files/upload",
        files={"file": ("part.stl", io.BytesIO(MINIMAL_STL), "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )
    file_id = up.json()["id"]
    resp = client.get(f"/files/{file_id}", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_get_file_not_found():
    token = _get_token()
    resp = client.get("/files/99999", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 404


def test_delete_file_soft():
    token = _get_token()
    up = client.post(
        "/files/upload",
        files={"file": ("part.stl", io.BytesIO(MINIMAL_STL), "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )
    file_id = up.json()["id"]
    resp = client.delete(f"/files/{file_id}", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 204
    # Should no longer appear in list
    list_resp = client.get("/files/", headers={"Authorization": f"Bearer {token}"})
    assert all(f["id"] != file_id for f in list_resp.json())
    # Direct get should also return 404
    get_resp = client.get(f"/files/{file_id}", headers={"Authorization": f"Bearer {token}"})
    assert get_resp.status_code == 404


def test_unauthenticated_upload():
    resp = client.post(
        "/files/upload",
        files={"file": ("part.stl", io.BytesIO(MINIMAL_STL), "application/octet-stream")},
    )
    assert resp.status_code == 401
