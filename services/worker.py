"""
Celery application and background task definitions.

If CELERY_BROKER_URL is not set, tasks are executed eagerly (in-process)
so the app continues to work without a worker.

To run a worker:
    celery -A services.worker worker --loglevel=info

Supported brokers (set CELERY_BROKER_URL in .env):
    redis://localhost:6379/0     — recommended
    amqp://guest@localhost//     — RabbitMQ
"""
import logging
import os

from celery import Celery

logger = logging.getLogger(__name__)

BROKER_URL = os.getenv("CELERY_BROKER_URL", "")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", BROKER_URL or "cache+memory://")

celery_app = Celery(
    "cadfactory",
    broker=BROKER_URL or "memory://",
    backend=BACKEND_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # When no broker is configured, run tasks synchronously in the calling process.
    task_always_eager=not bool(BROKER_URL),
    task_eager_propagates=True,
)

if not BROKER_URL:
    logger.debug(
        "CELERY_BROKER_URL not set — Celery tasks will run synchronously (no worker needed). "
        "Set CELERY_BROKER_URL=redis://localhost:6379/0 to enable async workers."
    )
else:
    logger.info(f"Celery broker: {BROKER_URL}")


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

@celery_app.task(name="cadfactory.tasks.run_topology_optimization", bind=True, max_retries=1)
def run_topology_optimization_task(self, geom_dict: dict, volfrac: float = 0.4, penal: float = 3.0):
    """
    Run SIMP topology optimization in a background worker.

    Args:
        geom_dict: Serialized geometry data (volume_cm3, bounding_box_mm, complexity_score).
        volfrac:   Target volume fraction (0–1).
        penal:     SIMP penalization exponent.

    Returns:
        dict with 'success', 'result' (density array encoded as list), and optional 'error'.
    """
    try:
        from services.topology_opt import run_simp_optimization
        from types import SimpleNamespace

        geom = SimpleNamespace(**geom_dict)
        result = run_simp_optimization(geom, volfrac=volfrac, penal=penal)
        return {"success": True, "result": result}
    except Exception as exc:
        logger.error(f"Topology optimization task failed: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=5)


@celery_app.task(name="cadfactory.tasks.run_gemini_fine_tune", bind=True, max_retries=0)
def run_gemini_fine_tune_task(self, training_data: list, display_name: str):
    """
    Submit a Gemini supervised fine-tuning job in a background worker.

    Args:
        training_data: List of {"input": str, "output": str} examples.
        display_name:  Human-readable name for the tuning job.

    Returns:
        dict with 'success', 'job_name', and optional 'error'.
    """
    try:
        from services.fine_tuner import submit_fine_tune_job
        job_name = submit_fine_tune_job(training_data, display_name)
        return {"success": True, "job_name": job_name}
    except Exception as exc:
        logger.error(f"Fine-tune task failed: {exc}", exc_info=True)
        return {"success": False, "error": str(exc)}


@celery_app.task(name="cadfactory.tasks.cleanup_generated_files", bind=True)
def cleanup_generated_files_task(self, max_age_hours: int = 24):
    """
    Periodic cleanup of old generated STL/STEP files.
    Intended to be scheduled via Celery Beat.
    """
    try:
        from services.cadquery_runner import cleanup_old_files
        cleanup_old_files(max_age_hours=max_age_hours)
        return {"success": True}
    except Exception as exc:
        logger.error(f"File cleanup task failed: {exc}", exc_info=True)
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Celery Beat schedule (optional — requires celery[redis] + celery beat)
# ---------------------------------------------------------------------------

celery_app.conf.beat_schedule = {
    "cleanup-generated-files-every-6-hours": {
        "task": "cadfactory.tasks.cleanup_generated_files",
        "schedule": 6 * 3600,  # seconds
        "kwargs": {"max_age_hours": 24},
    },
}
