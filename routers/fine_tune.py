"""
Fine-tuning management router.

POST /api/fine-tune/start              — Start a new job from thumbs-up examples
GET  /api/fine-tune/jobs               — List all jobs + their states
POST /api/fine-tune/jobs/{id}/refresh  — Pull latest state from Gemini API
POST /api/fine-tune/jobs/{id}/activate — Use this model for all future generations
POST /api/fine-tune/deactivate         — Switch back to base model
GET  /api/fine-tune/active             — Which model is currently active?
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import models
import security as _sec
from database import get_db
from services import fine_tuner

get_current_user = _sec.get_current_user
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fine-tune", tags=["Fine-tuning"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class StartTuningRequest(BaseModel):
    display_name: str = Field(default="cadfactory-tuned", description="Human-readable label")
    epoch_count: int = Field(default=5, ge=1, le=20)
    batch_size: int = Field(default=4, ge=1, le=16)
    learning_rate_multiplier: float = Field(default=1.0, gt=0.0)


class TuningJobOut(BaseModel):
    id: int
    display_name: str
    job_name: str
    state: str
    base_model: str
    training_examples: int
    is_active: bool
    tuned_model_name: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class ActiveModelOut(BaseModel):
    active: bool
    tuned_model_name: Optional[str] = None
    display_name: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_to_out(job: models.TunedModel) -> TuningJobOut:
    return TuningJobOut(
        id=job.id,
        display_name=job.display_name,
        job_name=job.job_name,
        state=job.state,
        base_model=job.base_model,
        training_examples=job.training_examples,
        is_active=job.is_active,
        tuned_model_name=job.tuned_model_name,
        error_message=job.error_message,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/start", response_model=TuningJobOut)
def start_fine_tuning(
    req: StartTuningRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Start a supervised fine-tuning job using all thumbs-up feedback examples.

    Requires at least MIN_EXAMPLES (10) positive examples in the database.
    The job runs asynchronously on Gemini infrastructure (~1-2 hours).
    Poll status via POST /api/fine-tune/jobs/{id}/refresh.
    """
    feedbacks = (
        db.query(models.GenerationFeedback)
        .filter(models.GenerationFeedback.rating == 1)
        .all()
    )
    examples = fine_tuner.prepare_training_data(feedbacks)

    if len(examples) < fine_tuner.MIN_EXAMPLES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Need at least {fine_tuner.MIN_EXAMPLES} thumbs-up examples, "
                f"found {len(examples)}. "
                "Thumbs-up more generated parts to build your training set."
            ),
        )

    try:
        result = fine_tuner.start_tuning_job(
            examples=examples,
            display_name=req.display_name,
            epoch_count=req.epoch_count,
            batch_size=req.batch_size,
            learning_rate_multiplier=req.learning_rate_multiplier,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini tuning API error: {e}")

    job = models.TunedModel(
        display_name=req.display_name,
        job_name=result["job_name"],
        base_model=fine_tuner.TUNING_BASE_MODEL,
        state=result["state"],
        training_examples=len(examples),
        is_active=False,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    logger.info(
        f"User {current_user.email} started tuning job {job.job_name} "
        f"with {len(examples)} examples"
    )
    return _job_to_out(job)


@router.get("/jobs", response_model=List[TuningJobOut])
def list_tuning_jobs(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """List all fine-tuning jobs, newest first."""
    jobs = (
        db.query(models.TunedModel)
        .order_by(models.TunedModel.created_at.desc())
        .all()
    )
    return [_job_to_out(j) for j in jobs]


@router.post("/jobs/{job_id}/refresh", response_model=TuningJobOut)
def refresh_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Pull the latest state of a tuning job from the Gemini API and update the DB.
    Call this periodically to check if training has finished.
    """
    job = db.query(models.TunedModel).filter(models.TunedModel.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Tuning job not found")

    try:
        status = fine_tuner.refresh_job_status(job.job_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

    job.state = status["state"]
    if status["tuned_model_name"]:
        job.tuned_model_name = status["tuned_model_name"]
    if status.get("error"):
        job.error_message = str(status["error"])
    if "SUCCEEDED" in job.state and job.completed_at is None:
        job.completed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(job)
    return _job_to_out(job)


@router.post("/jobs/{job_id}/activate", response_model=TuningJobOut)
def activate_model(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Mark this tuned model as the active model for all future generations.
    Job must be in SUCCEEDED state with a tuned_model_name available.
    """
    job = db.query(models.TunedModel).filter(models.TunedModel.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Tuning job not found")
    if "SUCCEEDED" not in job.state:
        raise HTTPException(
            status_code=422,
            detail=f"Job is not complete yet (state: {job.state}). Refresh first."
        )
    if not job.tuned_model_name:
        raise HTTPException(
            status_code=422,
            detail="No tuned model name available — call /refresh first to sync from Gemini."
        )

    # Deactivate any previously active model
    db.query(models.TunedModel).filter(
        models.TunedModel.is_active.is_(True)
    ).update({"is_active": False})

    job.is_active = True
    db.commit()
    db.refresh(job)

    logger.info(
        f"User {current_user.email} activated tuned model: {job.tuned_model_name}"
    )
    return _job_to_out(job)


@router.post("/deactivate")
def deactivate_all(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Deactivate all tuned models — generation will use the default base model."""
    updated = (
        db.query(models.TunedModel)
        .filter(models.TunedModel.is_active.is_(True))
        .update({"is_active": False})
    )
    db.commit()
    return {
        "deactivated": updated,
        "message": "Switched back to base model (gemini-2.5-pro / gemini-2.5-flash).",
    }


@router.get("/active", response_model=ActiveModelOut)
def get_active_model(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Return the currently active tuned model, or indicate that the base model is in use."""
    row = (
        db.query(models.TunedModel)
        .filter(
            models.TunedModel.is_active.is_(True),
            models.TunedModel.tuned_model_name.isnot(None),
        )
        .first()
    )
    if row:
        return ActiveModelOut(
            active=True,
            tuned_model_name=row.tuned_model_name,
            display_name=row.display_name,
            message=f"Using tuned model '{row.display_name}' trained on {row.training_examples} examples.",
        )
    return ActiveModelOut(
        active=False,
        message="Using base model (gemini-2.5-pro + gemini-2.5-flash two-pass pipeline).",
    )
