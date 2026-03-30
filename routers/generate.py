"""
/api/generate router — assembled from focused sub-modules.

Sub-modules:
  generate_core.py    — POST / /precision /stream /from-step/stream
  generate_refine.py  — POST /refine /refine/stream /execute /suggest-loads /decompose
  generate_history.py — GET/DELETE /history /history/{id}, GET /files/{filename}, POST /multiview
  generate_image.py   — POST /from-image /from-image/stream
  generate_schemas.py — shared Pydantic models + DB helpers (no routes)
"""

from fastapi import APIRouter

from routers.generate_core import router as core_router
from routers.generate_history import router as history_router
from routers.generate_image import router as image_router
from routers.generate_refine import router as refine_router

router = APIRouter(prefix="/api/generate", tags=["generate"])
router.include_router(core_router)
router.include_router(refine_router)
router.include_router(history_router)
router.include_router(image_router)
