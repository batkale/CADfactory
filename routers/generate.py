"""
FastAPI router for AI-powered CAD generation.
Thin layer: validates requests, delegates to services, returns responses.

Location: cadfactory-backend/routers/generate.py

Endpoints:
    POST /api/generate/          — Generate a part from natural language
    POST /api/generate/stream    — Same, but streams progress via SSE
    POST /api/generate/refine    — Refine an existing generated part
    POST /api/generate/execute   — Run a CadQuery script directly (no AI)
    POST /api/generate/suggest-loads — Get AI-suggested load cases for TO
    GET  /api/generate/history   — List user's generated parts
    GET  /api/generate/history/{part_id} — Get one generated part
    DELETE /api/generate/history/{part_id} — Delete a generated part
    GET  /api/generate/files/{filename}  — Serve generated STL/STEP files
"""

import os
import asyncio
import json as _json
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import security as auth_utils
get_current_user = auth_utils.get_current_user

from config.prompts import (
    DESIGN_PLAN_PROMPT, DESIGN_PLAN_TO_CODE_TEMPLATE,
    RETRY_PROMPT_TEMPLATE, build_system_prompt, build_system_prompt_with_examples,
    STEP_IMPORT_SYSTEM_PROMPT, REFINE_SYSTEM_PROMPT,
    SHAPE_RESEARCH_PROMPT_TO_PLAN, SHAPE_RESEARCH_INJECTION,
)
from services.claude_cad import (
    generate_and_execute, refine_script, suggest_load_cases,
    _generate_content, MODEL_FLASH, MODEL_PRO, check_complexity, MAX_RETRIES,
    generate_with_tuned_model, research_shape,
)
from services.semantic_decomposer import decompose_prompt, DecomposedObject
from services.csg_builder import build_from_prompt_result
from services.templates import TEMPLATE_MAP
from services.rag_store import get_similar_examples, store_count
from services.cadquery_runner import get_file_url, OUTPUT_DIR, execute_cadquery_sandboxed, EXECUTION_TIMEOUT
from services.script_utils import (
    validate_script, extract_bom_from_script,
    parse_json_response, extract_python_code,
)
from services.fine_tuner import get_active_tuned_model
from database import get_db, SessionLocal
import models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/generate", tags=["generate"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Request body for part generation."""
    description: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        examples=["Servo mounting bracket for MG996R with 4x M3 bolt holes"]
    )
    manufacturing_method: str = Field(
        default="fdm",
        pattern="^(fdm|sla|sls|cnc|sheet_metal|injection)$",
        description="Target manufacturing method"
    )
    constraints: Optional[dict] = Field(
        default=None,
        examples=[{"width": 40, "height": 30, "wall_thickness": 3}]
    )
    force: bool = Field(
        default=False,
        description="Force generation even when the prompt is vague"
    )


class PartMetadata(BaseModel):
    """Metadata for a single component in an assembly."""
    name: str = Field(..., description="Semantic name of the component")
    stl_url: str = Field(..., description="URL to download the component STL")
    color: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response body for part generation."""
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
    # CSG pipeline metadata (backwards-compatible — all optional)
    pipeline: Optional[str] = None           # "csg_builder" | "ai_fallback" | "needs_clarification" | "precision_8layer"
    confidence: Optional[float] = None       # decomposer confidence score
    object_name: Optional[str] = None        # recognised object name
    dims_estimated: Optional[bool] = None    # whether dimensions were estimated
    # 8-Layer precision pipeline report (only present when precision pipeline is used)
    pipeline_report: Optional[dict] = None   # PipelineResult.to_dict() output


class RefineRequest(BaseModel):
    """Request body for script refinement."""
    message: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["Make the walls thicker", "Add a cable routing slot on the back"]
    )
    current_script: str = Field(
        ...,
        min_length=10,
        description="The CadQuery script to modify"
    )
    manufacturing_method: str = Field(default="fdm")
    conversation_history: Optional[list] = Field(
        default=None,
        description="Previous messages for multi-turn refinement"
    )


class ExecuteRequest(BaseModel):
    """Request body for direct script execution (no AI)."""
    script: str = Field(..., min_length=10, description="CadQuery Python script to execute")
    manufacturing_method: str = Field(default="fdm")


class SuggestLoadsRequest(BaseModel):
    """Request body for AI-suggested load cases."""
    description: str = Field(
        ...,
        examples=["This bracket holds a NEMA 17 stepper motor to an aluminium extrusion"]
    )
    bbox_x: float = Field(..., description="Bounding box X dimension in mm")
    bbox_y: float = Field(..., description="Bounding box Y dimension in mm")
    bbox_z: float = Field(..., description="Bounding box Z dimension in mm")
    volume: float = Field(..., description="Part volume in mm³")
    surface_area: float = Field(default=0.0, description="Part surface area in mm²")


class GeneratedPartSummary(BaseModel):
    """Summary of a saved generated part (for history list)."""
    id: int
    description: str
    manufacturing_method: str
    attempts: int
    generation_time_s: float
    created_at: str
    stl_url: Optional[str] = None
    step_url: Optional[str] = None


class GeneratedPartDetail(GeneratedPartSummary):
    """Full detail of a saved generated part (includes script + BOM)."""
    script: str
    bom_suggestion: list
    warnings: list


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
    """Save a successfully generated part to the DB. Returns the new part ID."""
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=GenerateResponse)
async def generate_part(req: GenerateRequest, user=Depends(get_current_user)):
    """
    Generate a 3D part from a natural language description.
    Uses two-pass Gemini pipeline: Flash for design plan, Pro for code.
    Retries up to 3 times with error feedback if execution fails.
    """
    logger.info(
        f"Generate request from {getattr(user, 'email', '?')}: "
        f"{req.description[:60] if len(req.description) > 60 else req.description}... ({req.manufacturing_method})"  # type: ignore
    )

    result = await generate_and_execute(
        description=req.description,
        manufacturing_method=req.manufacturing_method,
        constraints=req.constraints,
    )

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail={
                "error": result.error,
                "script": result.script,
                "warnings": result.warnings,
                "attempts": result.attempts,
            }
        )

    part_id = _save_generated_part(
        user_id=user.id,
        description=req.description,
        manufacturing_method=req.manufacturing_method,
        script=result.script,
        stl_path=result.stl_path,
        step_path=result.step_path,
        bom=result.bom_suggestion,
        warnings=result.warnings,
        attempts=result.attempts,
        generation_time_s=round(result.total_time_s, 2),
    )

    return GenerateResponse(
        success=True,
        stl_url=get_file_url(result.stl_path) if result.stl_path else None,
        step_url=get_file_url(result.step_path) if result.step_path else None,
        script=result.script,
        parts=[
            PartMetadata(name=p["name"], stl_url=get_file_url(p["stl_path"]))
            for p in (result.parts or [])
        ],
        bom_suggestion=result.bom_suggestion,
        warnings=result.warnings,
        attempts=result.attempts,
        generation_time_s=round(result.total_time_s, 2),
        part_id=part_id,
    )


# ---------------------------------------------------------------------------
# 8-Layer Precision Pipeline endpoint
# ---------------------------------------------------------------------------

@router.post("/precision", response_model=GenerateResponse)
async def generate_part_precision(
    req: GenerateRequest,
    user=Depends(get_current_user),
    skip_layer8: bool = Query(
        default=False,
        description="Skip the Layer 8 AI script overhaul (faster but less detailed)"
    ),
):
    """
    Generate a 3D part using the full 8-layer precision pipeline.

    Layers:
      1. NLP Contextual Extraction (AI)
      2. Parameterization & Sorting (algorithmic)
      3. Mathematical Constraint Validation (deterministic math)
      4. Generative Gap-Filling (AI)
      5. CSG Topology Planning (algorithmic)
      6. Edge-Case Guard (AI)
      7. Granular Script Generation (deterministic, 5-10x detail)
      8. Script Linting & Precision Overhaul (AI — Pro model)

    Response includes `pipeline_report` with per-layer timing and confidence scores.
    """
    import time as _time
    from services.pipeline import run_precision_pipeline

    logger.info(
        f"[Precision Pipeline] request from {getattr(user, 'email', '?')}: "
        f"{req.description[:60] if len(req.description) > 60 else req.description}... ({req.manufacturing_method})"  # type: ignore
    )

    # Run the 8-layer pipeline
    pipeline_result = await run_precision_pipeline(
        prompt=req.description,
        manufacturing_method=req.manufacturing_method,
        constraints=req.constraints,
        skip_layer8=skip_layer8,
    )

    final_script = pipeline_result.final_script

    if not pipeline_result.success or not final_script:
        raise HTTPException(
            status_code=422,
            detail={
                "error": pipeline_result.error or "Precision pipeline failed",
                "pipeline_report": pipeline_result.to_dict(),
            }
        )

    # Execute the generated script
    from services.cadquery_runner import execute_cadquery_sandboxed
    from services.script_utils import validate_script, extract_bom_from_script

    is_valid, val_warnings = validate_script(final_script)
    if not is_valid:
        # One final self-heal attempt via the legacy retry system
        logger.warning("[Precision] script failed linting — one legacy retry")
        from services.claude_cad import generate_and_execute
        legacy = await generate_and_execute(
            req.description, req.manufacturing_method, req.constraints
        )
        if not legacy.success:
            raise HTTPException(
                status_code=422,
                detail={"error": "Script failed validation and legacy fallback also failed.",
                        "pipeline_report": pipeline_result.to_dict()}
            )
        # Use legacy result
        return GenerateResponse(
            success=True,
            script=legacy.script or "",
            stl_url=get_file_url(legacy.stl_path) if legacy.stl_path else None,
            step_url=get_file_url(legacy.step_path) if legacy.step_path else None,
            bom_suggestion=legacy.bom_suggestion,
            warnings=val_warnings + (legacy.warnings or []),
            attempts=legacy.attempts,
            generation_time_s=pipeline_result.total_elapsed_s,
            pipeline="precision_8layer_legacy_fallback",
            confidence=pipeline_result.overall_accuracy,
            object_name=pipeline_result.object_name,
            pipeline_report=pipeline_result.to_dict(),
        )

    exec_result = execute_cadquery_sandboxed(final_script)

    all_warnings = list(val_warnings)
    attempts = 1

    # Self-heal retry: if execution fails, fall back to legacy pipeline
    if not exec_result.success:
        logger.warning(
            f"[Precision] Layer 7/8 script failed execution — "
            f"falling back to legacy pipeline: {exec_result.error[:120] if exec_result.error else '?'}"
        )
        from services.claude_cad import generate_and_execute
        legacy = await generate_and_execute(
            req.description, req.manufacturing_method, req.constraints
        )
        if legacy.success:
            final_script = legacy.script or ""
            exec_result = type("R", (), {
                "success": True,
                "stl_path": legacy.stl_path,
                "step_path": legacy.step_path,
                "error": None,
            })()
            attempts = legacy.attempts
            all_warnings.append(
                f"Precision script fell back to legacy pipeline after execution failure "
                f"({attempts} attempt(s))"
            )
        else:
            exec_result = legacy  # still failed — error will be caught below

    if not exec_result.success:
        raise HTTPException(
            status_code=422,
            detail={
                "error": exec_result.error,
                "script": final_script,
                "pipeline_report": pipeline_result.to_dict(),
            }
        )

    bom = extract_bom_from_script(final_script)
    part_id = _save_generated_part(
        user_id=user.id,
        description=req.description,
        manufacturing_method=req.manufacturing_method,
        script=final_script,
        stl_path=exec_result.stl_path,
        step_path=exec_result.step_path,
        bom=bom,
        warnings=all_warnings,
        attempts=attempts,
        generation_time_s=pipeline_result.total_elapsed_s,
    )

    return GenerateResponse(
        success=True,
        stl_url=get_file_url(exec_result.stl_path) if exec_result.stl_path else None,
        step_url=get_file_url(exec_result.step_path) if exec_result.step_path else None,
        script=final_script,
        bom_suggestion=bom,
        warnings=all_warnings,
        attempts=attempts,
        generation_time_s=pipeline_result.total_elapsed_s,
        part_id=part_id,
        pipeline="precision_8layer",
        confidence=pipeline_result.overall_accuracy,
        object_name=pipeline_result.object_name,
        dims_estimated=False,  # precision pipeline always derives full dims
        pipeline_report=pipeline_result.to_dict(),
    )


@router.post("/stream")
async def generate_stream(req: GenerateRequest, user=Depends(get_current_user)):
    """
    Same as POST /api/generate/ but streams progress via Server-Sent Events.

    Event types:
      event: progress  — {"stage": str, "label": str, "done": bool}
      event: result    — GenerateResponse JSON (final, success or failure)

    Stages (in order): understanding → planning → generating → executing → exporting
    """
    user_id = user.id

    async def event_gen():
        import time

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

        # Complexity guard
        complexity_issue = check_complexity(req.description)
        if complexity_issue:
            yield sse("result", {"success": False, "error": complexity_issue})
            return

        start_time = time.time()
        all_warnings: list = []

        # ── Step 0: RAG (fetch similar approved examples) ─────────────────────
        rag_examples = []
        try:
            rag_examples = get_similar_examples(req.description, req.manufacturing_method, k=3)
            if rag_examples:
                logger.info(f"RAG: injecting {len(rag_examples)} similar example(s) into stream prompt")
        except Exception as e:
            logger.warning(f"RAG lookup failed in stream (non-fatal): {e}")

        # ── Stage 1: Understanding + Decompose ───────────────────────────────
        yield sse("progress", {"stage": "understanding",
                                "label": "Analysing your description...",
                                "done": False})

        decomposed = await asyncio.to_thread(decompose_prompt, req.description, rag_examples=rag_examples)

        yield sse("progress", {"stage": "understanding",
                                "label": "Understanding your description...",
                                "done": True})

        # If the AI is unsure, warn but allow generation to continue
        if decomposed.clarification_needed and decomposed.confidence < 0.5:
            if not req.force:
                yield sse("clarification", {
                    "confidence": decomposed.confidence,
                    "object_name": decomposed.object_name,
                    "question": decomposed.clarification_question or "Could you describe the shape in more detail?",
                })
                return
            else:
                all_warnings.append(f"Weak prompt (confidence {decomposed.confidence:.0%}) — generating anyway")

        # Try CSG builder path if confidence is sufficient
        csg_result = None
        if decomposed.confidence >= 0.5 and decomposed.operations:
            csg_result = build_from_prompt_result(decomposed)
            if not csg_result["success"]:
                logger.info(f"CSG builder validation failed: {csg_result['errors']} — falling back to AI")
                csg_result = None

        # ── Stage 1.5: Template Path (HIGHEST PRIORITY) ───────────────────
        template_script = None
        template_errors = []
        if decomposed.template_name in TEMPLATE_MAP:
            from services.templates import validate_template_params
            template_errors = validate_template_params(decomposed.template_name, decomposed.template_params)
            
            if not template_errors:
                try:
                    template_fn = TEMPLATE_MAP[decomposed.template_name]
                    template_script = template_fn(decomposed.template_params)
                    logger.info(f"Using template: {decomposed.template_name}")
                except Exception as e:
                    logger.warning(f"Template generation failed: {e}")
                    template_errors = [f"Template generation failed: {e}"]
            else:
                logger.info(f"Template validation failed: {template_errors} — falling back to AI")

        if template_script or csg_result:
            # ── Template or CSG fast path ─────────────────────────────────
            yield sse("progress", {"stage": "executing",
                                    "label": "Building 3D model from verified path..." if template_script else "Building 3D model from geometry tree...",
                                    "done": False})

            final_script = template_script or csg_result["script"]
            all_warnings.extend(template_errors)
            is_valid, validation_warnings = validate_script(final_script)
            all_warnings.extend(validation_warnings)

            exec_result = await asyncio.to_thread(execute_cadquery_sandboxed, final_script)
            attempts = 1

            if exec_result.success:
                yield sse("progress", {"stage": "exporting", "label": "Exporting STL...", "done": True})
                bom = extract_bom_from_script(final_script)
                part_id = _save_generated_part(
                    user_id=user_id,
                    description=req.description,
                    manufacturing_method=req.manufacturing_method,
                    script=final_script,
                    stl_path=exec_result.stl_path,
                    step_path=exec_result.step_path,
                    bom=bom,
                    warnings=all_warnings + (csg_result.get("errors", []) if csg_result else []),
                    attempts=1,
                    generation_time_s=round(time.time() - start_time, 2),
                )
                yield sse("result", {
                    "success": True,
                    "stl_url": get_file_url(exec_result.stl_path) if exec_result.stl_path else None,
                    "step_url": get_file_url(exec_result.step_path) if exec_result.step_path else None,
                    "script": final_script,
                    "parts": [
                        {"name": p["name"], "stl_url": get_file_url(p["stl_path"])}
                        for p in (exec_result.parts or [])
                    ],
                    "bom_suggestion": bom,
                    "warnings": all_warnings,
                    "attempts": 1,
                    "generation_time_s": round(time.time() - start_time, 2),
                    "part_id": part_id,
                    "pipeline": "template" if template_script else "csg_builder",
                    "confidence": decomposed.confidence,
                    "object_name": decomposed.object_name,
                    "dims_estimated": decomposed.dims_are_estimated,
                })
                return

            # Fast path execution failed — fall through to AI pipeline
            logger.info(f"Fast path script execution failed ({exec_result.error[:100]}) — falling back to AI")

        # ── Tuned model fast path (if a fine-tuned model is active) ───────────
        tuned_model_name = get_active_tuned_model(SessionLocal())
        if tuned_model_name:
            yield sse("progress", {"stage": "generating",
                                    "label": "Generating script with fine-tuned model...",
                                    "done": False})
            tuned_result = await asyncio.to_thread(
                generate_with_tuned_model,
                tuned_model_name, req.description, req.manufacturing_method,
            )
            if tuned_result.success:
                yield sse("progress", {"stage": "executing",
                                        "label": "Executing fine-tuned script...",
                                        "done": False})
                exec_result = await asyncio.to_thread(
                    execute_cadquery_sandboxed, tuned_result.script
                )
                if exec_result.success:
                    yield sse("progress", {"stage": "exporting", "label": "Exporting STL...", "done": True})
                    bom = extract_bom_from_script(tuned_result.script)
                    part_id = _save_generated_part(
                        user_id=user_id,
                        description=req.description,
                        manufacturing_method=req.manufacturing_method,
                        script=tuned_result.script,
                        stl_path=exec_result.stl_path,
                        step_path=exec_result.step_path,
                        bom=bom,
                        warnings=all_warnings,
                        attempts=1,
                        generation_time_s=round(time.time() - start_time, 2),
                    )
                    yield sse("result", {
                        "success": True,
                        "stl_url": get_file_url(exec_result.stl_path) if exec_result.stl_path else None,
                        "step_url": get_file_url(exec_result.step_path) if exec_result.step_path else None,
                        "script": tuned_result.script,
                        "bom_suggestion": bom,
                        "warnings": all_warnings,
                        "attempts": 1,
                        "generation_time_s": round(time.time() - start_time, 2),
                        "part_id": part_id,
                        "pipeline": "tuned_model",
                        "confidence": decomposed.confidence,
                        "object_name": decomposed.object_name,
                        "dims_estimated": decomposed.dims_are_estimated,
                    })
                    return
                logger.info(f"Tuned model script failed execution — falling back to base AI pipeline")
            else:
                logger.info(f"Tuned model generation failed — falling back to base AI pipeline")

        # ── Stage 1.9: Shape Research — AI describes the standard form ─────────
        yield sse("progress", {"stage": "planning",
                                "label": "Researching object shape...",
                                "done": False})

        shape_research = None
        try:
            shape_research = await asyncio.to_thread(research_shape, req.description)
            if shape_research:
                logger.info(f"Shape research: {shape_research[:120]}...")
        except Exception as e:
            logger.warning(f"Shape research failed (non-fatal): {e}")

        # ── Stage 2: Planning — Flash builds a JSON design plan ───────────────
        yield sse("progress", {"stage": "planning",
                                "label": "Designing geometry...",
                                "done": False})

        plan_json = None
        plan_user_msg = f"Part description: {req.description}"
        if req.constraints:
            cl = [f"  - {k}: {v}" for k, v in req.constraints.items()]
            plan_user_msg += "\n\nDimensional constraints:\n" + "\n".join(cl)
        plan_user_msg += f"\n\nManufacturing method: {req.manufacturing_method}"

        try:
            # Use shape-research-aware planning prompt if research succeeded
            planning_prompt = DESIGN_PLAN_PROMPT
            if shape_research:
                planning_prompt = SHAPE_RESEARCH_PROMPT_TO_PLAN.format(
                    shape_research=shape_research
                )

            raw_plan = await asyncio.to_thread(
                _generate_content, MODEL_FLASH, planning_prompt, plan_user_msg
            )
            plan_json = parse_json_response(raw_plan)
        except Exception as e:
            logger.warning(f"Planning pass failed: {e}")

        yield sse("progress", {"stage": "planning",
                                "label": "Designing geometry...",
                                "done": True})

        # ── Stage 3: Code generation — Pro writes CadQuery from the plan ──────
        yield sse("progress", {"stage": "generating",
                                "label": "Generating CadQuery script...",
                                "done": False})

        # ── Stage 4: Prompt Construction with RAG ───────────────────────────
        system_prompt = build_system_prompt_with_examples(req.manufacturing_method, rag_examples)

        if plan_json:
            constraints_str = ""
            if req.constraints:
                cl = [f"  - {k}: {v}" for k, v in req.constraints.items()]
                constraints_str = "\n\nDimensional constraints:\n" + "\n".join(cl)
            user_message = DESIGN_PLAN_TO_CODE_TEMPLATE.format(
                plan=_json.dumps(plan_json, indent=2),
                manufacturing_method=req.manufacturing_method,
                constraints=constraints_str,
            )
            # Inject shape research so code gen knows what the object looks like
            if shape_research:
                user_message += "\n\n" + SHAPE_RESEARCH_INJECTION.format(
                    shape_research=shape_research
                )
        else:
            user_message = f"Generate a CadQuery script for: {req.description}"
            if shape_research:
                user_message += "\n\n" + SHAPE_RESEARCH_INJECTION.format(
                    shape_research=shape_research
                )
            if req.constraints:
                cl = [f"  - {k}: {v}" for k, v in req.constraints.items()]
                user_message += "\n\nDimensional constraints:\n" + "\n".join(cl)
            user_message += f"\n\nManufacturing method: {req.manufacturing_method}"

        try:
            raw_text = await asyncio.to_thread(
                _generate_content, MODEL_PRO, system_prompt, user_message
            )
            script = extract_python_code(raw_text)
        except Exception as e:
            yield sse("result", {"success": False,
                                  "error": f"Failed to generate script: {str(e)}"})
            return

        yield sse("progress", {"stage": "generating",
                                "label": "Generating CadQuery script...",
                                "done": True})

        # Validate script safety
        is_valid, validation_warnings = validate_script(script)
        all_warnings.extend(validation_warnings)
        if not is_valid:
            yield sse("result", {
                "success": False,
                "error": "Script failed safety validation",
                "warnings": all_warnings,
                "script": script,
            })
            return

        # ── Stage 4: Execute ──────────────────────────────────────────────────
        yield sse("progress", {"stage": "executing",
                                "label": "Building 3D model...",
                                "done": False})

        result = await asyncio.to_thread(execute_cadquery_sandboxed, script)
        attempts = 1

        # Self-healing retry loop
        if not result.success:
            current_error = result.error
            for attempt in range(MAX_RETRIES):
                attempt_num = attempt + 2
                yield sse("progress", {
                    "stage": "executing",
                    "label": f"Fixing errors (attempt {attempt_num})...",
                    "done": False,
                })
                retry_msg = RETRY_PROMPT_TEMPLATE.format(
                    error=current_error, script=script
                )
                try:
                    raw_retry = await asyncio.to_thread(
                        _generate_content, MODEL_FLASH, system_prompt, retry_msg
                    )
                    script = extract_python_code(raw_retry)
                    result = await asyncio.to_thread(execute_cadquery_sandboxed, script)
                    attempts = attempt_num
                    if result.success:
                        all_warnings.append(
                            f"Script required {attempts} attempt(s) to succeed"
                        )
                        break
                    current_error = result.error
                except Exception:
                    continue

        total_time = time.time() - start_time

        if not result.success:
            yield sse("result", {
                "success": False,
                "error": result.error,
                "script": script,
                "warnings": all_warnings,
                "attempts": attempts,
                "generation_time_s": round(total_time, 2),
            })
            return

        # ── Stage 5: Export complete ──────────────────────────────────────────
        yield sse("progress", {"stage": "exporting",
                                "label": "Exporting STL...",
                                "done": True})

        bom = extract_bom_from_script(script)

        # Save to database
        part_id = _save_generated_part(
            user_id=user_id,
            description=req.description,
            manufacturing_method=req.manufacturing_method,
            script=script,
            stl_path=result.stl_path,
            step_path=result.step_path,
            bom=bom,
            warnings=all_warnings,
            attempts=attempts,
            generation_time_s=round(total_time, 2),
        )

        yield sse("result", {
            "success": True,
            "stl_url": get_file_url(result.stl_path) if result.stl_path else None,
            "step_url": get_file_url(result.step_path) if result.step_path else None,
            "script": script,
            "parts": [
                {"name": p["name"], "stl_url": get_file_url(p["stl_path"])}
                for p in (result.parts or [])
            ],
            "bom_suggestion": bom,
            "warnings": all_warnings,
            "attempts": attempts,
            "generation_time_s": round(total_time, 2),
            "part_id": part_id,
            "pipeline": "ai_fallback",
            "confidence": decomposed.confidence,
            "object_name": decomposed.object_name,
            "dims_estimated": decomposed.dims_are_estimated,
        })

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


class FromStepRequest(BaseModel):
    """Request body for generating a part that references an uploaded STEP file."""
    file_id: int = Field(..., description="UploadedFile ID of the reference STEP")
    description: str = Field(..., min_length=5, max_length=2000,
                              description="What to create / how to modify the STEP")
    manufacturing_method: str = Field(default="fdm",
                                       pattern="^(fdm|sla|sls|cnc|sheet_metal|injection)$")


@router.post("/from-step/stream")
async def generate_from_step_stream(req: FromStepRequest, user=Depends(get_current_user)):
    """
    Generate or modify a part that references an uploaded STEP file, streaming progress via SSE.

    The STEP file is copied into the CadQuery sandbox so the generated script can
    `cq.importers.importStep('uploaded_part.step')` without filesystem access concerns.
    """
    from database import SessionLocal

    db = SessionLocal()
    try:
        step_file = db.query(models.UploadedFile).filter(
            models.UploadedFile.id == req.file_id,
            models.UploadedFile.user_id == user.id,
        ).first()
        if not step_file:
            raise HTTPException(status_code=404, detail="STEP file not found")
        if step_file.file_format not in ("STEP",):
            raise HTTPException(status_code=422, detail="File must be a STEP file")
        step_path = step_file.upload_path
    finally:
        db.close()

    if not os.path.exists(step_path):
        raise HTTPException(status_code=404, detail="STEP file not found on disk")

    user_id = user.id

    async def event_gen():
        import time

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

        start_time = time.time()
        all_warnings: list = []

        yield sse("progress", {"stage": "understanding",
                                "label": "Reading STEP reference...", "done": True})

        yield sse("progress", {"stage": "planning",
                                "label": "Planning modifications...", "done": False})

        # Single-pass generation using the STEP-aware system prompt
        system_prompt = STEP_IMPORT_SYSTEM_PROMPT
        user_message = (
            f"Reference STEP file: uploaded_part.step\n\n"
            f"Task: {req.description}\n\n"
            f"Manufacturing method: {req.manufacturing_method}\n\n"
            "Write a CadQuery script that imports the STEP file and performs the task."
        )

        yield sse("progress", {"stage": "planning", "label": "Planning modifications...", "done": True})

        yield sse("progress", {"stage": "generating",
                                "label": "Generating CadQuery script...", "done": False})

        try:
            raw_text = await asyncio.to_thread(
                _generate_content, MODEL_PRO, system_prompt, user_message
            )
            script = extract_python_code(raw_text)
        except Exception as e:
            yield sse("result", {"success": False, "error": f"Failed to generate script: {e}"})
            return

        yield sse("progress", {"stage": "generating",
                                "label": "Generating CadQuery script...", "done": True})

        is_valid, validation_warnings = validate_script(script)
        all_warnings.extend(validation_warnings)
        if not is_valid:
            yield sse("result", {"success": False, "error": "Script failed safety validation",
                                  "warnings": all_warnings, "script": script})
            return

        yield sse("progress", {"stage": "executing",
                                "label": "Building 3D model...", "done": False})

        extra_files = {"uploaded_part.step": step_path}
        result = await asyncio.to_thread(
            execute_cadquery_sandboxed, script, EXECUTION_TIMEOUT, True, extra_files
        )
        attempts = 1

        if not result.success:
            current_error = result.error
            for attempt in range(MAX_RETRIES):
                attempt_num = attempt + 2
                yield sse("progress", {"stage": "executing",
                                        "label": f"Fixing errors (attempt {attempt_num})...",
                                        "done": False})
                retry_msg = RETRY_PROMPT_TEMPLATE.format(error=current_error, script=script)
                try:
                    raw_retry = await asyncio.to_thread(
                        _generate_content, MODEL_FLASH, system_prompt, retry_msg
                    )
                    script = extract_python_code(raw_retry)
                    result = await asyncio.to_thread(
                        execute_cadquery_sandboxed, script, EXECUTION_TIMEOUT, True, extra_files
                    )
                    attempts = attempt_num
                    if result.success:
                        all_warnings.append(f"Script required {attempts} attempt(s) to succeed")
                        break
                    current_error = result.error
                except Exception:
                    continue

        total_time = time.time() - start_time

        if not result.success:
            yield sse("result", {"success": False, "error": result.error, "script": script,
                                  "warnings": all_warnings, "attempts": attempts,
                                  "generation_time_s": round(total_time, 2)})
            return

        yield sse("progress", {"stage": "exporting", "label": "Exporting STL...", "done": True})

        bom = extract_bom_from_script(script)
        part_id = _save_generated_part(
            user_id=user_id,
            description=f"[STEP-based] {req.description}",
            manufacturing_method=req.manufacturing_method,
            script=script,
            stl_path=result.stl_path,
            step_path=result.step_path,
            bom=bom,
            warnings=all_warnings,
            attempts=attempts,
            generation_time_s=round(total_time, 2),
        )

        yield sse("result", {
            "success": True,
            "stl_url": get_file_url(result.stl_path) if result.stl_path else None,
            "step_url": get_file_url(result.step_path) if result.step_path else None,
            "script": script,
            "bom_suggestion": bom,
            "warnings": all_warnings,
            "attempts": attempts,
            "generation_time_s": round(total_time, 2),
            "part_id": part_id,
        })

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/refine", response_model=GenerateResponse)
async def refine_part(req: RefineRequest, user=Depends(get_current_user)):
    """
    Refine an existing generated part via natural language.

    Takes the current CadQuery script and a modification request,
    returns the updated script and new STL/STEP files.
    """
    logger.info(
        f"Refine request from {getattr(user, 'email', '?')}: "
        f"{req.message[:60]}..."
    )

    result = await refine_script(
        current_script=req.current_script,
        user_message=req.message,
        conversation_history=req.conversation_history,
        manufacturing_method=req.manufacturing_method,
    )

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail={
                "error": result.error,
                "script": result.script,
                "warnings": result.warnings,
            }
        )

    part_id = _save_generated_part(
        user_id=user.id,
        description=req.message,
        manufacturing_method=req.manufacturing_method,
        script=result.script,
        stl_path=result.stl_path,
        step_path=result.step_path,
        bom=result.bom_suggestion,
        warnings=result.warnings,
        attempts=result.attempts,
        generation_time_s=round(result.total_time_s, 2),
    )

    return GenerateResponse(
        success=True,
        stl_url=get_file_url(result.stl_path) if result.stl_path else None,
        step_url=get_file_url(result.step_path) if result.step_path else None,
        script=result.script,
        bom_suggestion=result.bom_suggestion,
        warnings=result.warnings,
        generation_time_s=round(result.total_time_s, 2),
        part_id=part_id,
    )


@router.post("/refine/stream")
async def refine_stream(req: RefineRequest, user=Depends(get_current_user)):
    """
    Streaming version of /refine.  Uses SSE with the same 5-stage UI as /stream.
    Uses Gemini Flash (faster) with a surgical-edit prompt.
    """
    user_id = user.id

    async def event_gen():
        import time
        start_time = time.time()

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

        all_warnings: list = []

        # Stage 1: understanding
        yield sse("progress", {"stage": "understanding",
                                "label": "Analysing modification request...",
                                "done": False})
        yield sse("progress", {"stage": "understanding",
                                "label": "Analysing modification request...",
                                "done": True})

        # Stage 2: generating — Flash applies minimal edits
        yield sse("progress", {"stage": "generating",
                                "label": "Applying changes to script...",
                                "done": False})

        refine_msg = (
            f"Current script:\n```python\n{req.current_script}\n```\n\n"
            f"Modification request: {req.message}"
        )

        try:
            raw_text = await asyncio.to_thread(
                _generate_content, MODEL_FLASH, REFINE_SYSTEM_PROMPT, refine_msg
            )
            script = extract_python_code(raw_text)
        except Exception as e:
            yield sse("result", {"success": False, "error": f"Failed to modify script: {e}"})
            return

        yield sse("progress", {"stage": "generating",
                                "label": "Applying changes to script...",
                                "done": True})

        # Validate
        is_valid, validation_warnings = validate_script(script)
        all_warnings.extend(validation_warnings)
        if not is_valid:
            yield sse("result", {"success": False,
                                  "error": "Script failed safety validation",
                                  "script": script, "warnings": all_warnings})
            return

        # Stage 3: executing
        yield sse("progress", {"stage": "executing",
                                "label": "Building 3D model...",
                                "done": False})

        result = await asyncio.to_thread(execute_cadquery_sandboxed, script)
        attempts = 1

        # Self-healing retry loop (same as generate)
        if not result.success:
            system_prompt = build_system_prompt(req.manufacturing_method)
            current_error = result.error
            for attempt in range(MAX_RETRIES):
                attempt_num = attempt + 2
                yield sse("progress", {
                    "stage": "executing",
                    "label": f"Fixing errors (attempt {attempt_num})...",
                    "done": False,
                })
                retry_msg = RETRY_PROMPT_TEMPLATE.format(error=current_error, script=script)
                try:
                    raw_retry = await asyncio.to_thread(
                        _generate_content, MODEL_FLASH, system_prompt, retry_msg
                    )
                    script = extract_python_code(raw_retry)
                    result = await asyncio.to_thread(execute_cadquery_sandboxed, script)
                    attempts = attempt_num
                    if result.success:
                        all_warnings.append(f"Script required {attempts} attempt(s) to succeed")
                        break
                    current_error = result.error
                except Exception:
                    continue

        total_time = time.time() - start_time

        if not result.success:
            yield sse("result", {
                "success": False,
                "error": result.error,
                "script": script,
                "warnings": all_warnings,
                "attempts": attempts,
                "generation_time_s": round(total_time, 2),
            })
            return

        # Stage 4: export complete
        yield sse("progress", {"stage": "exporting", "label": "Exporting STL...", "done": True})

        bom = extract_bom_from_script(script)
        part_id = _save_generated_part(
            user_id=user_id,
            description=req.message,
            manufacturing_method=req.manufacturing_method,
            script=script,
            stl_path=result.stl_path,
            step_path=result.step_path,
            bom=bom,
            warnings=all_warnings,
            attempts=attempts,
            generation_time_s=round(total_time, 2),
        )

        yield sse("result", {
            "success": True,
            "stl_url": get_file_url(result.stl_path) if result.stl_path else None,
            "step_url": get_file_url(result.step_path) if result.step_path else None,
            "script": script,
            "bom_suggestion": bom,
            "warnings": all_warnings,
            "attempts": attempts,
            "generation_time_s": round(total_time, 2),
            "part_id": part_id,
        })

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/execute", response_model=GenerateResponse)
async def execute_script(req: ExecuteRequest, user=Depends(get_current_user)):
    """
    Execute a CadQuery script directly without AI generation.
    Used for re-running user-edited scripts.
    """
    is_valid, validation_warnings = validate_script(req.script)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={"error": "Script failed safety validation", "warnings": validation_warnings}
        )

    result = await asyncio.to_thread(execute_cadquery_sandboxed, req.script)
    if not result.success:
        raise HTTPException(
            status_code=422,
            detail={"error": result.error, "script": req.script, "warnings": validation_warnings}
        )

    bom = extract_bom_from_script(req.script)
    return GenerateResponse(
        success=True,
        stl_url=get_file_url(result.stl_path) if result.stl_path else None,
        step_url=get_file_url(result.step_path) if result.step_path else None,
        script=req.script,
        bom_suggestion=bom,
        warnings=validation_warnings,
        generation_time_s=round(result.execution_time_s, 2),
    )


@router.post("/suggest-loads")
async def get_load_suggestions(req: SuggestLoadsRequest, user=Depends(get_current_user)):
    """
    Get AI-suggested load cases for topology optimization.
    """
    suggestions = await suggest_load_cases(
        description=req.description,
        bbox=(req.bbox_x, req.bbox_y, req.bbox_z),
        volume=req.volume,
        surface_area=req.surface_area,
    )

    if not suggestions:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate load case suggestions"
        )

    return suggestions


# ---------------------------------------------------------------------------
# Decompose endpoint (debug / frontend preview)
# ---------------------------------------------------------------------------

class DecomposeRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)


@router.post("/decompose")
async def decompose_part_description(
    req: DecomposeRequest,
    user=Depends(get_current_user),
):
    """
    Run semantic decomposition on a prompt and return the CSG tree.
    Does NOT generate or execute any CadQuery — for debug/preview only.
    """
    result = await asyncio.to_thread(decompose_prompt, req.prompt)
    return result.model_dump()


# ---------------------------------------------------------------------------
# History endpoints
# ---------------------------------------------------------------------------

@router.get("/history", response_model=List[GeneratedPartSummary])
async def list_generated_history(
    limit: int = Query(default=50, ge=1, le=200),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all generated parts saved to the user's account, newest first."""
    parts = (
        db.query(models.GeneratedPart)
        .filter(models.GeneratedPart.user_id == user.id)
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
    """Delete a generated part from history."""
    part = (
        db.query(models.GeneratedPart)
        .filter(
            models.GeneratedPart.id == part_id,
            models.GeneratedPart.user_id == user.id,
        )
        .first()
    )
    if not part:
        raise HTTPException(status_code=404, detail="Generated part not found")
    db.delete(part)
    db.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# File serving
# ---------------------------------------------------------------------------

@router.get("/files/{filename}")
async def serve_generated_file(filename: str):
    """
    Serve a generated STL or STEP file.

    Security: only serves files from the OUTPUT_DIR, validates filename.
    """
    # Prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Only serve expected file types
    allowed_extensions = {".stl", ".step", ".stp", ".png", ".json"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File type {ext} not allowed")

    filepath = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    media_types = {
        ".stl": "application/sla",
        ".step": "application/step",
        ".stp": "application/step",
        ".png": "image/png",
        ".json": "application/json",
    }

    return FileResponse(
        filepath,
        media_type=media_types.get(ext, "application/octet-stream"),
        filename=filename
    )
