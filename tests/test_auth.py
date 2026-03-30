"""
Integration tests for /auth endpoints.
Uses TestClient with an in-memory SQLite database — no external services needed.
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use an in-memory database for tests
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_cadfactory.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-ci-only")

from database import Base, get_db  # noqa: E402
from main import app  # noqa: E402

TEST_DB_URL = "sqlite:///./test_cadfactory.db"
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
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app, raise_server_exceptions=False)

USER = {"email": "test@example.com", "username": "testuser", "password": "StrongPass1!"}


def test_register_success():
    resp = client.post("/auth/register", json=USER)
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_register_duplicate_email():
    client.post("/auth/register", json=USER)
    resp = client.post("/auth/register", json=USER)
    assert resp.status_code == 409


def test_register_duplicate_username():
    client.post("/auth/register", json=USER)
    resp = client.post("/auth/register", json={**USER, "email": "other@example.com"})
    assert resp.status_code == 409


def test_login_success():
    client.post("/auth/register", json=USER)
    resp = client.post("/auth/login", data={"username": USER["email"], "password": USER["password"]})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password():
    client.post("/auth/register", json=USER)
    resp = client.post("/auth/login", data={"username": USER["email"], "password": "wrongpass"})
    assert resp.status_code == 401


def test_login_unknown_user():
    resp = client.post("/auth/login", data={"username": "nobody@example.com", "password": "x"})
    assert resp.status_code == 401


def test_get_me_authenticated():
    reg = client.post("/auth/register", json=USER)
    token = reg.json()["access_token"]
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == USER["email"]


def test_get_me_unauthenticated():
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_delete_account():
    reg = client.post("/auth/register", json=USER)
    token = reg.json()["access_token"]
    resp = client.delete("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 204
    # Token should be invalid after soft-delete (is_active = False)
    resp2 = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp2.status_code == 401
