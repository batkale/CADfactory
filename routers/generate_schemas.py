"""
Shared Pydantic models and DB helpers for the generate sub-routers.
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import models
from database import SessionLocal
from services.cadquery_runner import get_file_url

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=2000,
                              examples=["Servo mounting bracket for MG996R with 4x M3 bolt holes"])
    manufacturing_method: str = Field(default="fdm",
                                      pattern="^(fdm|sla|sls|cnc|sheet_metal|injection)$")
    constraints: Optional[dict] = Field(default=None,
                                        examples=[{"width": 40, "height": 30, "wall_thickness": 3}])
    force: bool = Field(default=False)


class RefineRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=1000)
    current_script: str = Field(..., min_length=10)
    manufacturing_method: str = Field(default="fdm")
    conversation_history: Optional[list] = Field(default=None)


class ExecuteRequest(BaseModel):
    script: str = Field(..., min_length=10)
    manufacturing_method: str = Field(default="fdm")


class SuggestLoadsRequest(BaseModel):
    description: str = Field(..., examples=["Bracket holds NEMA 17 to aluminium extrusion"])
    bbox_x: float
    bbox_y: float
    bbox_z: float
    volume: float
    surface_area: float = Field(default=0.0)


class DecomposeRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)


class FromStepRequest(BaseModel):
    file_id: int = Field(..., description="UploadedFile ID of the reference STEP")
    description: str = Field(..., min_length=5, max_length=2000)
    manufacturing_method: str = Field(default="fdm",
                                      pattern="^(fdm|sla|sls|cnc|sheet_metal|injection)$")


class MultiviewRequest(BaseModel):
    stl_filename: str = Field(..., description="STL filename in generated_files/")
    width: int = Field(default=256, ge=64, le=1024)
    height: int = Field(default=256, ge=64, le=1024)


class ImageToCADRequest(BaseModel):
    additional_context: str = Field(default="", max_length=500)
    manufacturing_method: str = Field(default="fdm",
                                      pattern="^(fdm|sla|sls|cnc|sheet_metal|injection)$")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PartMetadata(BaseModel):
    name: str
    stl_url: str
    color: Optional[str] = None


class GenerateResponse(BaseModel):
    success: bool
    stl_url: Optional[str] = None
    step_url: Optional[str] = None
    script: str = ""
    parts: List[PartMetadata] = []
    bom_suggestion: list = []
    warnings: list = []
    error: Optional[str] = None
    attempts: int = 1
    generation_time_s: float = 0.0
    part_id: Optional[int] = None
    pipeline: Optional[str] = None
    confidence: Optional[float] = None
    object_name: Optional[str] = None
    dims_estimated: Optional[bool] = None
    pipeline_report: Optional[dict] = None
    design_notes: Optional[dict] = None


class GeneratedPartSummary(BaseModel):
    id: int
    description: str
    manufacturing_method: str
    attempts: int
    generation_time_s: float
    created_at: str
    stl_url: Optional[str] = None
    step_url: Optional[str] = None


class GeneratedPartDetail(GeneratedPartSummary):
    script: str
    bom_suggestion: list
    warnings: list


class ImageToCADResponse(BaseModel):
    extracted_description: str
    structured_plan: Optional[dict] = None
    confidence: float = 0.0
    message: str = ""


class MultiviewResponse(BaseModel):
    views: dict
    count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _part_to_summary(p: models.GeneratedPart) -> GeneratedPartSummary:
    return GeneratedPartSummary(
        id=p.id,
        description=p.description,
        manufacturing_method=p.manufacturing_method,
        attempts=p.attempts,
        generation_time_s=p.generation_time_s,
        created_at=p.created_at.isoformat(),
        stl_url=get_file_url(p.stl_path) if p.stl_path else None,
        step_url=get_file_url(p.step_path) if p.step_path else None,
    )


def _part_to_detail(p: models.GeneratedPart) -> GeneratedPartDetail:
    return GeneratedPartDetail(
        id=p.id,
        description=p.description,
        manufacturing_method=p.manufacturing_method,
        attempts=p.attempts,
        generation_time_s=p.generation_time_s,
        created_at=p.created_at.isoformat(),
        stl_url=get_file_url(p.stl_path) if p.stl_path else None,
        step_url=get_file_url(p.step_path) if p.step_path else None,
        script=p.script,
        bom_suggestion=p.bom_suggestion or [],
        warnings=p.warnings or [],
    )


def _save_generated_part(
    user_id: int,
    description: str,
    manufacturing_method: str,
    script: str,
    stl_path: Optional[str],
    step_path: Optional[str],
    bom: list,
    warnings: list,
    attempts: int,
    generation_time_s: float,
) -> Optional[int]:
    """Save a generated part to the DB. Returns the new part ID or None on failure."""
    db = SessionLocal()
    try:
        part = models.GeneratedPart(
            user_id=user_id,
            description=description,
            manufacturing_method=manufacturing_method,
            script=script,
            stl_path=stl_path,
            step_path=step_path,
            bom_suggestion=bom,
            warnings=warnings,
            attempts=attempts,
            generation_time_s=generation_time_s,
        )
        db.add(part)
        db.commit()
        db.refresh(part)
        return part.id
    except Exception as e:
        logger.error(f"Failed to save generated part: {e}")
        return None
    finally:
        db.close()
