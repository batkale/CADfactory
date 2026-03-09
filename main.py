import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from database import engine, Base
import models  # noqa: F401 — ensures models are registered before create_all
from routers import auth, files, analysis, materials

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all tables on startup
    Base.metadata.create_all(bind=engine)
    os.makedirs(os.getenv("UPLOAD_DIR", "./uploads"), exist_ok=True)
    print("✅ Database tables ready")
    print("✅ Upload directory ready")
    yield
    print("👋 Server shutting down")


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

# ── CORS ─────────────────────────────────────────────────────
# In production: replace "*" with your frontend domain
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
