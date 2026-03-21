"""
Feedback router — thumbs up / thumbs down on generated parts.

POST /api/feedback/          — submit a rating
GET  /api/feedback/{part_id} — get rating counts for a part
GET  /api/feedback/export    — export all positive examples (for fine-tuning)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import models
from database import get_db
import security as _sec
get_current_user = _sec.get_current_user
from services import rag_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class FeedbackIn(BaseModel):
    part_id: int
    rating: int = Field(..., description="+1 for thumbs up, -1 for thumbs down")
    comment: Optional[str] = None


class FeedbackOut(BaseModel):
    id: int
    part_id: int
    rating: int
    comment: Optional[str]
    message: str


class FeedbackStats(BaseModel):
    part_id: int
    thumbs_up: int
    thumbs_down: int
    user_rating: Optional[int]   # current user's rating, if any


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=FeedbackOut)
def submit_feedback(
    payload: FeedbackIn,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Submit or update a thumbs-up (+1) / thumbs-down (-1) rating for a generated part.

    Thumbs-up → script is added to the RAG store so future generations benefit.
    Thumbs-down → script is removed from the RAG store (if previously added).
    """
    if payload.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be +1 or -1")

    # Verify part exists and belongs to this user
    part = (
        db.query(models.GeneratedPart)
        .filter(
            models.GeneratedPart.id == payload.part_id,
            models.GeneratedPart.user_id == current_user.id,
        )
        .first()
    )
    if not part:
        raise HTTPException(status_code=404, detail="Part not found")

    # Upsert: one feedback record per user+part
    existing = (
        db.query(models.GenerationFeedback)
        .filter(
            models.GenerationFeedback.user_id == current_user.id,
            models.GenerationFeedback.part_id == payload.part_id,
        )
        .first()
    )

    if existing:
        existing.rating = payload.rating
        existing.comment = payload.comment
        existing.prompt_snapshot = part.description
        existing.script_snapshot = part.script
        existing.manufacturing_method = part.manufacturing_method
        fb = existing
    else:
        fb = models.GenerationFeedback(
            user_id=current_user.id,
            part_id=payload.part_id,
            rating=payload.rating,
            comment=payload.comment,
            prompt_snapshot=part.description,
            script_snapshot=part.script,
            manufacturing_method=part.manufacturing_method,
        )
        db.add(fb)

    db.commit()
    db.refresh(fb)

    # Sync to RAG store
    if payload.rating == 1:
        try:
            rag_store.add_example(
                part_id=part.id,
                description=part.description,
                script=part.script,
                manufacturing_method=part.manufacturing_method or "fdm",
            )
            msg = "Feedback saved — example added to RAG store for future generations."
        except Exception as e:
            logger.error(f"RAG add failed: {e}")
            msg = "Feedback saved (RAG store update failed — non-critical)."
    else:
        try:
            rag_store.remove_example(part_id=part.id)
            msg = "Feedback saved — example removed from RAG store."
        except Exception as e:
            logger.error(f"RAG remove failed: {e}")
            msg = "Feedback saved."

    return FeedbackOut(id=fb.id, part_id=fb.part_id, rating=fb.rating, comment=fb.comment, message=msg)


@router.get("/{part_id}", response_model=FeedbackStats)
def get_feedback_stats(
    part_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Return thumbs-up / thumbs-down counts for a part, plus the current user's vote."""
    rows = (
        db.query(models.GenerationFeedback)
        .filter(models.GenerationFeedback.part_id == part_id)
        .all()
    )
    thumbs_up = sum(1 for r in rows if r.rating == 1)
    thumbs_down = sum(1 for r in rows if r.rating == -1)
    user_rating = next(
        (r.rating for r in rows if r.user_id == current_user.id), None
    )
    return FeedbackStats(
        part_id=part_id,
        thumbs_up=thumbs_up,
        thumbs_down=thumbs_down,
        user_rating=user_rating,
    )


@router.get("/export/positive")
def export_positive_examples(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Export all thumbs-up feedback as (prompt, completion) pairs — ready for fine-tuning.
    Returns JSONL-style list of {"prompt": ..., "completion": ...} objects.
    """
    rows = (
        db.query(models.GenerationFeedback)
        .filter(models.GenerationFeedback.rating == 1)
        .all()
    )
    return [
        {
            "prompt": r.prompt_snapshot,
            "completion": r.script_snapshot,
            "manufacturing_method": r.manufacturing_method,
        }
        for r in rows
        if r.prompt_snapshot and r.script_snapshot
    ]
