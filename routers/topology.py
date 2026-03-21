"""
Topology optimization router.

Endpoints:
    POST /api/topology/optimize   — run SIMP on an uploaded STL
    GET  /api/topology/files/{fn} — serve the optimised STL
"""

import os
import uuid
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import json as _json

import security as auth_utils
get_current_user = auth_utils.get_current_user

from services import storage as storage_service
from services.cadquery_runner import OUTPUT_DIR   # reuse same output dir
from services.claude_cad import suggest_load_cases

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
                                        description="Part description — used to auto-set volfrac via AI")


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

def _get_output_path(suffix: str = ".stl") -> tuple[str, str]:
    """Return (absolute path, filename) for a new output file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fn = uuid.uuid4().hex[:12] + "_opt" + suffix
    return os.path.join(OUTPUT_DIR, fn), fn


_SAFE_VOLFRAC = 0.50   # conservative default when AI doesn't give a recommendation


async def _resolve_volfrac(req: OptimizeRequest) -> float:
    """
    If volfrac is explicitly set, use it.
    Otherwise call AI suggest_load_cases with the part description to get
    recommended_volume_fraction. Falls back to _SAFE_VOLFRAC on any failure.
    """
    if req.volfrac is not None:
        return req.volfrac

    if req.description:
        try:
            suggestion = await suggest_load_cases(
                description=req.description,
                bbox=(50.0, 50.0, 50.0),   # rough defaults — we don't have geometry yet
                volume=125_000.0,           # 5cm³ cube in mm³
                surface_area=15_000.0,
            )
            if suggestion:
                ai_vf = suggestion.get("recommended_volume_fraction")
                if isinstance(ai_vf, (int, float)) and 0.20 <= float(ai_vf) <= 0.80:
                    # Clamp to a safe range — never let AI go below 0.30 (too risky)
                    return max(0.30, float(ai_vf))
        except Exception as e:
            logger.warning(f"AI volfrac suggestion failed: {e}")

    return _SAFE_VOLFRAC


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/optimize/stream")
async def optimize_stream(req: OptimizeRequest, user=Depends(get_current_user)):
    """
    Run topology optimization and stream progress via SSE.

    Events:
      progress  — {"stage": str, "label": str, "pct": int}
      result    — OptimizeResponse JSON or {"error": str}
    """
    from database import SessionLocal
    import models

    # Resolve file path from DB
    db = SessionLocal()
    try:
        f = db.query(models.UploadedFile).filter(
            models.UploadedFile.id == req.file_id,
            models.UploadedFile.user_id == user.id,
        ).first()
        if not f:
            raise HTTPException(status_code=404, detail="File not found")
        stl_path   = f.upload_path
        file_format = f.file_format.upper()
    finally:
        db.close()

    if not os.path.exists(stl_path):
        raise HTTPException(status_code=404, detail="STL file not found on disk")
    if file_format not in ("STL", "3MF"):
        raise HTTPException(status_code=422, detail="Only STL/3MF files can be optimised")

    output_path, output_fn = _get_output_path()

    # Resolve volfrac (possibly via AI) before starting the SSE stream
    volfrac = await _resolve_volfrac(req)

    async def event_gen():
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

        yield sse("progress", {"stage": "voxelizing",
                                "label": f"Voxelizing mesh (target retention: {round(volfrac*100)}%)…",
                                "pct": 5})
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
                max_iter=60,          # more iterations → safer convergence, fewer empty-mesh errors
            )
        except Exception as e:
            logger.error(f"Topology optimization failed: {e}", exc_info=True)
            yield sse("result", {"error": str(e)})
            return

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
async def optimize_sync(req: OptimizeRequest, user=Depends(get_current_user)):
    """
    Blocking topology optimization endpoint (simpler alternative to /stream).
    May time out for high-resolution grids — prefer /optimize/stream.
    """
    from database import SessionLocal
    import models

    db = SessionLocal()
    try:
        f = db.query(models.UploadedFile).filter(
            models.UploadedFile.id == req.file_id,
            models.UploadedFile.user_id == user.id,
        ).first()
        if not f:
            raise HTTPException(status_code=404, detail="File not found")
        stl_path   = f.upload_path
        file_format = f.file_format.upper()
    finally:
        db.close()

    if not os.path.exists(stl_path):
        raise HTTPException(status_code=404, detail="STL file not found on disk")
    if file_format not in ("STL", "3MF"):
        raise HTTPException(status_code=422, detail="Only STL/3MF files can be optimised")

    output_path, output_fn = _get_output_path()

    try:
        from services.topology_opt import topology_optimize
        result = await asyncio.to_thread(
            topology_optimize,
            stl_path=stl_path,
            output_stl_path=output_path,
            volfrac=req.volfrac,
            resolution=req.resolution,
            fixed_side=req.fixed_side,
            load_side=req.load_side,
            load_dir=req.load_dir,
            material_key=req.material_key,
        )
    except Exception as e:
        logger.error(f"Topology optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return OptimizeResponse(
        **{k: v for k, v in result.items() if k != "output_stl_path"},
        optimized_stl_url=f"/api/topology/files/{output_fn}",
    )


@router.post("/suggest-loads")
async def suggest_loads(req: SuggestLoadsRequest, user=Depends(get_current_user)):
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
