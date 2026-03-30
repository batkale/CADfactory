"""
History, file-serving, and multiview endpoints for /api/generate.
"""
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

import models
import security as auth_utils
from database import get_db
from services.cadquery_runner import OUTPUT_DIR
from routers.generate_schemas import (
    GeneratedPartDetail,
    GeneratedPartSummary,
    MultiviewRequest,
    MultiviewResponse,
    _part_to_detail,
    _part_to_summary,
)

get_current_user = auth_utils.get_current_user
logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


@router.get("/history", response_model=List[GeneratedPartSummary])
async def list_generated_history(
    limit: int = Query(default=50, ge=1, le=200),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all generated parts saved to the user's account, newest first."""
    parts = (
        db.query(models.GeneratedPart)
        .filter(
            models.GeneratedPart.user_id == user.id,
            models.GeneratedPart.deleted_at.is_(None),
        )
        .order_by(models.GeneratedPart.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_part_to_summary(p) for p in parts]


@router.get("/history/{part_id}", response_model=GeneratedPartDetail)
async def get_generated_part(
    part_id: int,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get full detail (including script) for a single generated part."""
    part = (
        db.query(models.GeneratedPart)
        .filter(
            models.GeneratedPart.id == part_id,
            models.GeneratedPart.user_id == user.id,
            models.GeneratedPart.deleted_at.is_(None),
        )
        .first()
    )
    if not part:
        raise HTTPException(status_code=404, detail="Generated part not found")
    return _part_to_detail(part)


@router.delete("/history/{part_id}")
async def delete_generated_part(
    part_id: int,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Soft-delete a generated part from history."""
    part = (
        db.query(models.GeneratedPart)
        .filter(
            models.GeneratedPart.id == part_id,
            models.GeneratedPart.user_id == user.id,
            models.GeneratedPart.deleted_at.is_(None),
        )
        .first()
    )
    if not part:
        raise HTTPException(status_code=404, detail="Generated part not found")
    part.deleted_at = datetime.now(timezone.utc)
    db.commit()
    return {"ok": True}


@router.get("/files/{filename}")
async def serve_generated_file(filename: str):
    """
    Serve a generated STL or STEP file.
    Security: validates filename and ensures it resolves inside OUTPUT_DIR.
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    allowed_extensions = {".stl", ".step", ".stp", ".png", ".json", ".ppm"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File type {ext} not allowed")

    output_root = os.path.realpath(OUTPUT_DIR)
    filepath = os.path.realpath(os.path.join(OUTPUT_DIR, filename))
    if not filepath.startswith(output_root + os.sep) and filepath != output_root:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    media_types = {
        ".stl": "application/sla",
        ".step": "application/step",
        ".stp": "application/step",
        ".png": "image/png",
        ".json": "application/json",
        ".ppm": "image/x-portable-pixmap",
    }
    return FileResponse(
        filepath,
        media_type=media_types.get(ext, "application/octet-stream"),
        filename=filename,
    )


@router.post("/multiview", response_model=MultiviewResponse)
async def render_multiview_thumbnails(
    req: MultiviewRequest,
    current_user=Depends(get_current_user),
):
    """Render front/right/top/isometric thumbnails of a generated STL file."""
    from services.multiview_renderer import render_multiview

    if ".." in req.stl_filename or "/" in req.stl_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    stl_path = os.path.join(OUTPUT_DIR, req.stl_filename)
    if not os.path.exists(stl_path):
        raise HTTPException(status_code=404, detail="STL file not found")

    file_id = os.path.splitext(req.stl_filename)[0]
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: render_multiview(
                stl_path=stl_path,
                output_dir=OUTPUT_DIR,
                file_id=file_id,
                width=req.width,
                height=req.height,
            ),
        )
    except Exception as e:
        logger.error(f"Multiview rendering failed: {e}")
        raise HTTPException(status_code=500, detail="Rendering failed")

    view_urls = {
        view_name: f"/api/generate/files/{os.path.basename(path)}"
        for view_name, path in result.items()
    }
    return MultiviewResponse(views=view_urls, count=len(view_urls))
