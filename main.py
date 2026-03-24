import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from dotenv import load_dotenv

from database import engine, Base
import models  # noqa: F401 — ensures models are registered before create_all
from routers import auth, files, analysis, materials
from routers.generate import router as generate_router
from routers.topology import router as topology_router
from routers.feedback import router as feedback_router
from routers.fine_tune import router as fine_tune_router
from routers.search import router as search_router
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
        Base.metadata.create_all(bind=engine)
        logger.info("OK: Database tables ready")
    except Exception as e:
        logger.error(f"FAIL: Database init failed: {e}")
        raise
    try:
        upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"OK: Upload directory ready: {upload_dir}")
    except Exception as e:
        logger.error(f"FAIL: Upload dir failed: {e}")
    logger.info("=== Server ready: http://localhost:8000 ===")
    yield
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
    return response

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
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


@app.get("/health", tags=["Health"])
def health():
    gemini_key_set = bool(os.getenv("GEMINI_API_KEY", ""))
    return {
        "status": "ok",
        "gemini_configured": gemini_key_set,
        "database": os.getenv("DATABASE_URL", "sqlite:///./cadfactory.db"),
        "upload_dir": os.getenv("UPLOAD_DIR", "./uploads"),
    }
