import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import models  # noqa: F401 — ensures models are registered before create_all
from database import Base, engine
from routers import analysis, auth, files, materials
from routers.feedback import router as feedback_router
from routers.fine_tune import router as fine_tune_router
from routers.generate import router as generate_router
from routers.search import router as search_router
from routers.topology import router as topology_router

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cadfactory")
# Suppress noisy third-party loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== CADfactory starting up ===")
    try:
        from alembic import command as alembic_command
        from alembic.config import Config as AlembicConfig
        alembic_cfg = AlembicConfig("alembic.ini")
        alembic_command.upgrade(alembic_cfg, "head")
        logger.info("OK: Database migrations applied")
    except Exception as e:
        logger.warning(f"Alembic migration failed ({e}), falling back to create_all")
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("OK: Database tables ready (create_all fallback)")
        except Exception as e2:
            logger.error(f"FAIL: Database init failed: {e2}")
            raise
    try:
        upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"OK: Upload directory ready: {upload_dir}")
    except Exception as e:
        logger.error(f"FAIL: Upload dir failed: {e}")
    # Clean up old generated files on startup
    try:
        from services.cadquery_runner import cleanup_old_files
        cleanup_old_files(max_age_hours=24)
        logger.info("OK: Old generated files cleaned up")
    except Exception as e:
        logger.warning(f"File cleanup failed (non-fatal): {e}")

    # Background task: repeat cleanup every 6 hours
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(6 * 3600)
            try:
                from services.cadquery_runner import cleanup_old_files
                cleanup_old_files(max_age_hours=24)
            except Exception:
                pass

    cleanup_task = asyncio.create_task(periodic_cleanup())

    logger.info("=== Server ready: http://localhost:8000 ===")
    yield
    cleanup_task.cancel()
    logger.info("=== Server shutting down ===")


app = FastAPI(
    title="CADfactory API",
    description=(
        "Physics-to-capital compiler for hardware startups. "
        "Upload STL/STEP → geometry parsing → parametric COGS → AI analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Prometheus metrics (optional) ────────────────────────────
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response as _Response

    _http_requests_total = Counter(
        "cadfactory_http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )
    _http_request_duration_seconds = Histogram(
        "cadfactory_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "path"],
    )

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return _Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    _prometheus_enabled = True
    logger.info("OK: Prometheus metrics enabled at /metrics")
except ImportError:
    _prometheus_enabled = False
    logger.debug("prometheus_client not installed — /metrics endpoint disabled. "
                 "Install with: pip install prometheus_client")


# ── Request/response logger middleware ───────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(f"✗ UNHANDLED EXCEPTION on {request.method} {request.url.path}: {exc}", exc_info=True)
        raise
    elapsed = (time.time() - start) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(level, f"← {response.status_code} {request.method} {request.url.path} ({elapsed:.0f}ms)")

    # Record Prometheus metrics if available
    if _prometheus_enabled:
        _path = request.url.path
        _http_requests_total.labels(
            method=request.method, path=_path, status=response.status_code
        ).inc()
        _http_request_duration_seconds.labels(
            method=request.method, path=_path
        ).observe(elapsed / 1000)

    return response

# ── CORS ─────────────────────────────────────────────────────
_allowed_origins = os.getenv("CORS_ORIGINS", "").strip()
_origins = [o.strip() for o in _allowed_origins.split(",") if o.strip()] if _allowed_origins else ["*"]
if _origins == ["*"]:
    logger.warning(
        "CORS is open to all origins ('*'). "
        "Set CORS_ORIGINS in your .env file for production (e.g. https://yourdomain.com)."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiting ────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Routers ───────────────────────────────────────────────────
# All API routes are versioned under /v1.
# The unversioned paths are kept as aliases for backward compatibility.
v1 = APIRouter(prefix="/v1")
v1.include_router(auth.router)
v1.include_router(files.router)
v1.include_router(analysis.router)
v1.include_router(materials.router)
v1.include_router(generate_router)
v1.include_router(topology_router)
v1.include_router(feedback_router)
v1.include_router(fine_tune_router)
v1.include_router(search_router)
app.include_router(v1)

# Backward-compatible unversioned routes
app.include_router(auth.router)
app.include_router(files.router)
app.include_router(analysis.router)
app.include_router(materials.router)
app.include_router(generate_router)
app.include_router(topology_router)
app.include_router(feedback_router)
app.include_router(fine_tune_router)
app.include_router(search_router)

# ── Health check ──────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "CADfactory API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/app", tags=["Pages"], include_in_schema=False)
def app_page():
    return FileResponse("cadfactory.html", media_type="text/html")


@app.get("/viewer", tags=["Pages"], include_in_schema=False)
def viewer_page():
    return FileResponse("cadviewer.html", media_type="text/html")


@app.get("/health", tags=["Health"])
def health():
    gemini_key_set = bool(os.getenv("GEMINI_API_KEY", ""))
    return {
        "status": "ok",
        "gemini_configured": gemini_key_set,
        "database": os.getenv("DATABASE_URL", "sqlite:///./cadfactory.db"),
        "upload_dir": os.getenv("UPLOAD_DIR", "./uploads"),
    }
