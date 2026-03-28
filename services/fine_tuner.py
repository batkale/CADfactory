"""
Fine-tuning service for Gemini models.

Manages supervised fine-tuning jobs that learn from thumbs-up feedback.
A successfully trained model can replace the two-pass Flash+Pro pipeline,
generating CadQuery scripts in a single call with domain-specific knowledge.

Usage flow:
  1. Users thumbs-up good generations → positive examples accumulate in DB
  2. Admin calls POST /api/fine-tune/start → job submitted to Gemini
  3. Job runs async (~1-2 hours) → poll via POST /api/fine-tune/jobs/{id}/refresh
  4. On SUCCEEDED → POST /api/fine-tune/jobs/{id}/activate
  5. Generation pipeline uses tuned model instead of base MODEL_PRO
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from google import genai
from google.genai import types as genai_types
from sqlalchemy.orm import Session

import models

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# gemini-1.5-flash-001-tuning is currently the only Gemini model that supports
# supervised fine-tuning via the public API.
TUNING_BASE_MODEL = os.environ.get(
    "GEMINI_TUNING_BASE_MODEL",
    "models/gemini-1.5-flash-001-tuning",
)

MIN_EXAMPLES = 10


def _get_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------

def prepare_training_data(
    feedbacks: list,
) -> List[genai_types.TuningExample]:
    """
    Convert positive-feedback DB rows to TuningExample objects.

    Each example teaches the model: given this description, produce this script.
    We include the manufacturing method in the input so the model learns to
    tailor geometry for FDM vs CNC vs injection moulding etc.
    """
    examples: List[genai_types.TuningExample] = []
    for fb in feedbacks:
        if not fb.prompt_snapshot or not fb.script_snapshot:
            continue
        method = fb.manufacturing_method or "fdm"
        text_input = (
            f"Generate a CadQuery Python script for the following 3D part.\n"
            f"Description: {fb.prompt_snapshot}\n"
            f"Manufacturing method: {method}\n\n"
            f"Return only executable Python code. "
            f"Use show_object(result) at the end."
        )
        examples.append(
            genai_types.TuningExample(
                text_input=text_input,
                output=fb.script_snapshot,
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

def start_tuning_job(
    examples: List[genai_types.TuningExample],
    display_name: str = "cadfactory-tuned",
    epoch_count: int = 5,
    batch_size: int = 4,
    learning_rate_multiplier: float = 1.0,
) -> dict:
    """
    Submit a supervised fine-tuning job to Gemini.

    Returns dict with job_name (Gemini resource name) and initial state string.
    The job runs asynchronously; call refresh_job_status() to poll progress.
    """
    client = _get_client()
    job = client.tunings.tune(
        base_model=TUNING_BASE_MODEL,
        training_dataset=genai_types.TuningDataset(
            examples=genai_types.TuningExamples(examples=examples)
        ),
        config=genai_types.CreateTuningJobConfig(
            tuned_model_display_name=display_name,
            epoch_count=epoch_count,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier,
        ),
    )
    logger.info(f"Submitted tuning job: {job.name} (state={job.state})")
    return {
        "job_name": job.name,
        "state": str(job.state),
        "tuned_model_name": None,
    }


def refresh_job_status(job_name: str) -> dict:
    """
    Poll Gemini for the current state of a tuning job.

    Returns dict with keys:
      state            — string representation of JobState enum
      tuned_model_name — resource name usable for generate_content (or None)
      error            — error string if FAILED, else None
    """
    client = _get_client()
    job = client.tunings.get(name=job_name)

    tuned_model_name: Optional[str] = None
    if job.tuned_model and hasattr(job.tuned_model, "model"):
        tuned_model_name = job.tuned_model.model

    error = None
    if hasattr(job, "error") and job.error:
        error = str(job.error)

    return {
        "state": str(job.state),
        "tuned_model_name": tuned_model_name,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Active model lookup (called by generate.py at request time)
# ---------------------------------------------------------------------------

def get_active_tuned_model(db: Session) -> Optional[str]:
    """
    Return the tuned model resource name if one is marked active and succeeded.
    Returns None if no tuned model is active (generation uses base model).
    """
    row = (
        db.query(models.TunedModel)
        .filter(
            models.TunedModel.is_active.is_(True),
            models.TunedModel.state.contains("SUCCEEDED"),
            models.TunedModel.tuned_model_name.isnot(None),
        )
        .first()
    )
    return row.tuned_model_name if row else None
