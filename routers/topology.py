"""
Topology optimization router.

Endpoints:
    POST /api/topology/optimize/stream — run SIMP on an uploaded STL, SSE progress
    POST /api/topology/optimize        — blocking version (prefer /stream)
    POST /api/topology/suggest-loads   — AI boundary condition suggestion
    GET  /api/topology/files/{fn}      — serve the optimised STL
"""

import asyncio
import json as _json
import logging
import os
import queue
import threading
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import models
import security as auth_utils
from database import get_db
from services.cadquery_runner import OUTPUT_DIR
from services.claude_cad import suggest_load_cases

get_current_user = auth_utils.get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/topology", tags=["topology"])

# ── Pydantic models ───────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    file_id: int = Field(..., description="UploadedFile ID to optimize")
    volfrac: Optional[float] = Field(
        default=None, ge=0.10, le=0.90,
        description="Target material retention (None = let AI decide; default safe: 0.50)",
    )
    resolution: int = Field(default=20, ge=8, le=40,
                             description="Voxel resolution (higher = slower, more detail)")
    fixed_side: str = Field(default="bottom",
                             pattern="^(bottom|top|left|right|front|back)$")
    load_side: str = Field(default="top",
                            pattern="^(bottom|top|left|right|front|back)$")
    load_dir: int = Field(default=2, ge=0, le=2,
                           description="Force direction: 0=x, 1=y, 2=z")
    material_key: str = Field(default="pla",
                               description="Material key for mass/cost calc")
    description: Optional[str] = Field(default=None,
                                        description="Part description for AI-based volfrac/BC suggestion")


class OptimizeResponse(BaseModel):
    original_volume_cm3: float
    optimized_volume_cm3: float
    volume_saved_pct: float
    original_mass_g: float
    optimized_mass_g: float
    mass_saved_g: float
    mass_saved_pct: float
    cost_saved_per_unit_usd: float
    print_time_saved_min: float
    material: str
    material_key: str
    volfrac: float
    fixed_side: str
    load_side: str
    optimized_stl_url: str


class SuggestLoadsRequest(BaseModel):
    description: str
    bbox_x: float
    bbox_y: float
    bbox_z: float
    volume: float
    surface_area: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

_SAFE_VOLFRAC = 0.50


def _get_output_path(suffix: str = ".stl") -> tuple[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fn = uuid.uuid4().hex[:12] + "_opt" + suffix
    return os.path.join(OUTPUT_DIR, fn), fn


def _get_file_or_404(file_id: int, user_id: int, db: Session) -> models.UploadedFile:
    """Fetch an active UploadedFile belonging to the user, or raise 404."""
    f = (
        db.query(models.UploadedFile)
        .filter(
            models.UploadedFile.id == file_id,
            models.UploadedFile.user_id == user_id,
            models.UploadedFile.deleted_at.is_(None),
        )
        .first()
    )
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    if not os.path.exists(f.upload_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    if f.file_format.upper() not in ("STL", "3MF"):
        raise HTTPException(status_code=422, detail="Only STL/3MF files can be optimised")
    return f


async def _resolve_volfrac(req: OptimizeRequest, stl_path: str) -> float:
    """
    Return the volfrac to use.

    Priority:
      1. Explicitly set in the request → use as-is.
      2. Description provided → ask AI using real file geometry.
      3. Fallback to _SAFE_VOLFRAC.
    """
    if req.volfrac is not None:
        return req.volfrac

    if req.description:
        try:
            from services.topology_opt import load_stl_triangles
            import numpy as np
            tris = load_stl_triangles(stl_path)
            verts = tris.reshape(-1, 3)
            mn, mx = verts.min(axis=0), verts.max(axis=0)
            extents = mx - mn                         # mm
            bbox_mm3 = float(np.prod(extents))        # bounding box volume in mm³
            vol_mm3  = float(bbox_mm3 * 0.6)          # rough fill estimate

            suggestion = await suggest_load_cases(
                description=req.description,
                bbox=(float(extents[0]), float(extents[1]), float(extents[2])),
                volume=vol_mm3,
                surface_area=0.0,
            )
            if suggestion:
                ai_vf = suggestion.get("recommended_volume_fraction")
                if isinstance(ai_vf, (int, float)) and 0.20 <= float(ai_vf) <= 0.80:
                    return max(0.30, float(ai_vf))
        except Exception as e:
            logger.warning(f"AI volfrac suggestion failed: {e}")

    return _SAFE_VOLFRAC


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/optimize/stream")
async def optimize_stream(
    req: OptimizeRequest,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Run topology optimization and stream real-time progress via SSE.

    Events:
      progress  — {"stage": str, "label": str, "pct": int}
      result    — OptimizeResponse JSON or {"error": str}
    """
    f = _get_file_or_404(req.file_id, user.id, db)
    stl_path = f.upload_path

    output_path, output_fn = _get_output_path()

    # Resolve volfrac using real file geometry before opening the stream
    volfrac = await _resolve_volfrac(req, stl_path)
    max_iter = max(40, int(req.resolution * 4.2))

    async def event_gen():
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

        yield sse("progress", {
            "stage": "voxelizing",
            "label": f"Voxelizing mesh (target retention: {round(volfrac * 100)}%)…",
            "pct": 5,
        })

        # Progress events come from the optimizer thread via a thread-safe queue.
        prog_queue: queue.SimpleQueue = queue.SimpleQueue()
        result_holder: dict = {}
        done_event = threading.Event()

        def run_optimizer():
            try:
                from services.topology_opt import topology_optimize

                def on_progress(iteration: int, total: int, obj: float, vol: float):
                    pct = 10 + int(85 * iteration / total)
                    prog_queue.put({
                        "stage": "simp",
                        "label": f"SIMP iteration {iteration}/{total} — vol {vol:.2f}",
                        "pct": pct,
                    })

                result_holder["result"] = topology_optimize(
                    stl_path=stl_path,
                    output_stl_path=output_path,
                    volfrac=volfrac,
                    resolution=req.resolution,
                    fixed_side=req.fixed_side,
                    load_side=req.load_side,
                    load_dir=req.load_dir,
                    material_key=req.material_key,
                    max_iter=max_iter,
                    progress_callback=on_progress,
                )
            except Exception as exc:
                result_holder["error"] = exc
            finally:
                done_event.set()

        thread = threading.Thread(target=run_optimizer, daemon=True)
        thread.start()

        # Drain progress events while the thread runs
        while not done_event.is_set():
            await asyncio.sleep(0.4)
            while not prog_queue.empty():
                yield sse("progress", prog_queue.get_nowait())

        # Drain any remaining events after thread finishes
        while not prog_queue.empty():
            yield sse("progress", prog_queue.get_nowait())

        if "error" in result_holder:
            logger.error(f"Topology optimization failed: {result_holder['error']}", exc_info=True)
            yield sse("result", {"error": str(result_holder["error"])})
            return

        result = result_holder["result"]
        yield sse("progress", {"stage": "done", "label": "Done!", "pct": 100})
        orig_fn = os.path.basename(result.get("original_stl_path", ""))
        yield sse("result", {
            **result,
            "volfrac_used": volfrac,
            "optimized_stl_url": f"/api/topology/files/{output_fn}",
            "original_stl_url":  f"/api/topology/files/{orig_fn}" if orig_fn else None,
        })

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_sync(
    req: OptimizeRequest,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Blocking topology optimization endpoint (prefer /optimize/stream for long runs).
    """
    f = _get_file_or_404(req.file_id, user.id, db)
    stl_path = f.upload_path

    output_path, output_fn = _get_output_path()
    volfrac = await _resolve_volfrac(req, stl_path)
    max_iter = max(40, int(req.resolution * 4.2))

    try:
        from services.topology_opt import topology_optimize
        result = await asyncio.to_thread(
            topology_optimize,
            stl_path=stl_path,
            output_stl_path=output_path,
            volfrac=volfrac,
            resolution=req.resolution,
            fixed_side=req.fixed_side,
            load_side=req.load_side,
            load_dir=req.load_dir,
            material_key=req.material_key,
            max_iter=max_iter,
        )
    except Exception as e:
        logger.error(f"Topology optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return OptimizeResponse(
        **{k: v for k, v in result.items() if k not in ("output_stl_path", "original_stl_path")},
        optimized_stl_url=f"/api/topology/files/{output_fn}",
    )


@router.post("/suggest-loads")
async def suggest_loads(
    req: SuggestLoadsRequest,
    user: models.User = Depends(get_current_user),
):
    """Use AI to suggest boundary conditions from part description + geometry."""
    suggestions = await suggest_load_cases(
        description=req.description,
        bbox=(req.bbox_x, req.bbox_y, req.bbox_z),
        volume=req.volume,
        surface_area=req.surface_area,
    )
    if not suggestions:
        raise HTTPException(status_code=500, detail="Failed to generate load suggestions")
    return suggestions


@router.api_route("/files/{filename}", methods=["GET", "HEAD"])
async def serve_optimized_file(filename: str):
    """Serve an optimized STL file."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filename.endswith(".stl"):
        raise HTTPException(status_code=400, detail="Only STL files served here")
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type="application/sla", filename=filename)
