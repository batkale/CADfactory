"""
Core generation endpoints for /api/generate:
  POST /          — generate from natural language
  POST /precision — 8-layer precision pipeline
  POST /stream    — streaming generation via SSE
  POST /from-step/stream — generation from a STEP reference file
"""
import asyncio
import json as _json
import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

import models
import security as auth_utils
from config.prompts import (
    DESIGN_PLAN_PROMPT,
    DESIGN_PLAN_TO_CODE_TEMPLATE,
    RETRY_PROMPT_TEMPLATE,
    SHAPE_RESEARCH_INJECTION,
    SHAPE_RESEARCH_PROMPT_TO_PLAN,
    STEP_IMPORT_SYSTEM_PROMPT,
    build_system_prompt_with_examples,
)
from database import SessionLocal, get_db
from services.cadquery_runner import EXECUTION_TIMEOUT, execute_cadquery_sandboxed, get_file_url
from services.claude_cad import (
    MAX_RETRIES,
    MODEL_FLASH,
    MODEL_PRO,
    _generate_content,
    check_complexity,
    generate_and_execute,
    generate_with_tuned_model,
    research_shape,
)
from services.csg_builder import build_from_prompt_result
from services.fine_tuner import get_active_tuned_model
from services.rag_store import get_similar_examples
from services.script_utils import (
    extract_bom_from_script,
    extract_python_code,
    parse_json_response,
    validate_script,
)
from services.semantic_decomposer import decompose_prompt
from services.templates import TEMPLATE_MAP
from routers.generate_schemas import (
    FromStepRequest,
    GenerateRequest,
    GenerateResponse,
    PartMetadata,
    _save_generated_part,
)

get_current_user = auth_utils.get_current_user
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


@router.post("/", response_model=GenerateResponse)
@limiter.limit("10/minute")
async def generate_part(request: Request, req: GenerateRequest, user=Depends(get_current_user)):
    """
    Generate a 3D part from a natural language description.
    Uses two-pass Gemini pipeline: Flash for design plan, Pro for code.
    Retries up to 3 times with error feedback if execution fails.
    """
    logger.info(
        f"Generate request from {getattr(user, 'email', '?')}: "
        f"{req.description[:60]}... ({req.manufacturing_method})"
    )

    result = await generate_and_execute(
        description=req.description,
        manufacturing_method=req.manufacturing_method,
        constraints=req.constraints,
    )

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail={"error": result.error, "script": result.script,
                    "warnings": result.warnings, "attempts": result.attempts},
        )

    part_id = _save_generated_part(
        user_id=user.id, description=req.description, manufacturing_method=req.manufacturing_method,
        script=result.script, stl_path=result.stl_path, step_path=result.step_path,
        bom=result.bom_suggestion, warnings=result.warnings,
        attempts=result.attempts, generation_time_s=round(result.total_time_s, 2),
    )

    return GenerateResponse(
        success=True,
        stl_url=get_file_url(result.stl_path) if result.stl_path else None,
        step_url=get_file_url(result.step_path) if result.step_path else None,
        script=result.script,
        parts=[PartMetadata(name=p["name"], stl_url=get_file_url(p["stl_path"]))
               for p in (result.parts or [])],
        bom_suggestion=result.bom_suggestion,
        warnings=result.warnings,
        attempts=result.attempts,
        generation_time_s=round(result.total_time_s, 2),
        part_id=part_id,
    )


@router.post("/precision", response_model=GenerateResponse)
@limiter.limit("10/minute")
async def generate_part_precision(
    request: Request,
    req: GenerateRequest,
    user=Depends(get_current_user),
    skip_layer8: bool = Query(default=False, description="Skip Layer 8 AI script overhaul"),
):
    """Generate a 3D part using the full 8-layer precision pipeline."""
    from services.pipeline import run_precision_pipeline

    logger.info(
        f"[Precision Pipeline] request from {getattr(user, 'email', '?')}: "
        f"{req.description[:60]}... ({req.manufacturing_method})"
    )

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
            detail={"error": pipeline_result.error or "Precision pipeline failed",
                    "pipeline_report": pipeline_result.to_dict()},
        )

    is_valid, val_warnings = validate_script(final_script)
    if not is_valid:
        logger.warning("[Precision] script failed linting — one legacy retry")
        legacy = await generate_and_execute(req.description, req.manufacturing_method, req.constraints)
        if not legacy.success:
            raise HTTPException(
                status_code=422,
                detail={"error": "Script failed validation and legacy fallback also failed.",
                        "pipeline_report": pipeline_result.to_dict()},
            )
        return GenerateResponse(
            success=True, script=legacy.script or "",
            stl_url=get_file_url(legacy.stl_path) if legacy.stl_path else None,
            step_url=get_file_url(legacy.step_path) if legacy.step_path else None,
            bom_suggestion=legacy.bom_suggestion,
            warnings=val_warnings + (legacy.warnings or []),
            attempts=legacy.attempts, generation_time_s=pipeline_result.total_elapsed_s,
            pipeline="precision_8layer_legacy_fallback",
            confidence=pipeline_result.overall_accuracy,
            object_name=pipeline_result.object_name,
            pipeline_report=pipeline_result.to_dict(),
        )

    exec_result = execute_cadquery_sandboxed(final_script)
    all_warnings = list(val_warnings)
    attempts = 1

    if not exec_result.success:
        logger.warning(f"[Precision] script failed execution — falling back to legacy pipeline")
        legacy = await generate_and_execute(req.description, req.manufacturing_method, req.constraints)
        if legacy.success:
            final_script = legacy.script or ""
            exec_result = type("R", (), {
                "success": True, "stl_path": legacy.stl_path,
                "step_path": legacy.step_path, "error": None,
            })()
            attempts = legacy.attempts
            all_warnings.append(f"Precision script fell back to legacy pipeline ({attempts} attempt(s))")
        else:
            exec_result = legacy

    if not exec_result.success:
        raise HTTPException(
            status_code=422,
            detail={"error": exec_result.error, "script": final_script,
                    "pipeline_report": pipeline_result.to_dict()},
        )

    bom = extract_bom_from_script(final_script)
    part_id = _save_generated_part(
        user_id=user.id, description=req.description, manufacturing_method=req.manufacturing_method,
        script=final_script, stl_path=exec_result.stl_path, step_path=exec_result.step_path,
        bom=bom, warnings=all_warnings, attempts=attempts, generation_time_s=pipeline_result.total_elapsed_s,
    )

    return GenerateResponse(
        success=True,
        stl_url=get_file_url(exec_result.stl_path) if exec_result.stl_path else None,
        step_url=get_file_url(exec_result.step_path) if exec_result.step_path else None,
        script=final_script, bom_suggestion=bom, warnings=all_warnings,
        attempts=attempts, generation_time_s=pipeline_result.total_elapsed_s,
        part_id=part_id, pipeline="precision_8layer",
        confidence=pipeline_result.overall_accuracy,
        object_name=pipeline_result.object_name, dims_estimated=False,
        pipeline_report=pipeline_result.to_dict(),
    )


@router.post("/stream")
@limiter.limit("10/minute")
async def generate_stream(request: Request, req: GenerateRequest, user=Depends(get_current_user)):
    """
    Stream generation progress via Server-Sent Events.
    Event types: progress, clarification, result.
    """
    user_id = user.id

    async def event_gen():
        import time

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

        complexity_issue = check_complexity(req.description)
        if complexity_issue:
            yield sse("result", {"success": False, "error": complexity_issue})
            return

        start_time = time.time()
        all_warnings: list = []

        # RAG lookup
        rag_examples = []
        try:
            rag_examples = get_similar_examples(req.description, req.manufacturing_method, k=3)
        except Exception as e:
            logger.warning(f"RAG lookup failed (non-fatal): {e}")

        yield sse("progress", {"stage": "understanding", "label": "Analysing your description...", "done": False})

        decomposed = await asyncio.to_thread(decompose_prompt, req.description, rag_examples=rag_examples)

        yield sse("progress", {"stage": "understanding", "label": "Understanding your description...", "done": True})

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

        # CSG builder path
        csg_result = None
        if decomposed.confidence >= 0.5 and decomposed.operations:
            csg_result = build_from_prompt_result(decomposed)
            if not csg_result["success"]:
                logger.info(f"CSG builder failed: {csg_result['errors']} — falling back to AI")
                csg_result = None

        # Template path (highest priority)
        template_script = None
        template_errors = []
        if decomposed.template_name in TEMPLATE_MAP:
            from services.templates import validate_template_params
            template_errors = validate_template_params(decomposed.template_name, decomposed.template_params)
            if not template_errors:
                try:
                    template_fn = TEMPLATE_MAP[decomposed.template_name]
                    template_script = template_fn(decomposed.template_params)
                except Exception as e:
                    template_errors = [f"Template generation failed: {e}"]
            else:
                logger.info(f"Template validation failed: {template_errors} — falling back to AI")

        if template_script or csg_result:
            yield sse("progress", {
                "stage": "executing",
                "label": "Building 3D model from verified path..." if template_script else "Building 3D model from geometry tree...",
                "done": False,
            })
            final_script = template_script or csg_result["script"]
            all_warnings.extend(template_errors)
            _, validation_warnings = validate_script(final_script)
            all_warnings.extend(validation_warnings)

            exec_result = await asyncio.to_thread(execute_cadquery_sandboxed, final_script)
            if exec_result.success:
                yield sse("progress", {"stage": "exporting", "label": "Exporting STL...", "done": True})
                bom = extract_bom_from_script(final_script)
                part_id = _save_generated_part(
                    user_id=user_id, description=req.description,
                    manufacturing_method=req.manufacturing_method, script=final_script,
                    stl_path=exec_result.stl_path, step_path=exec_result.step_path,
                    bom=bom, warnings=all_warnings + (csg_result.get("errors", []) if csg_result else []),
                    attempts=1, generation_time_s=round(time.time() - start_time, 2),
                )
                yield sse("result", {
                    "success": True,
                    "stl_url": get_file_url(exec_result.stl_path) if exec_result.stl_path else None,
                    "step_url": get_file_url(exec_result.step_path) if exec_result.step_path else None,
                    "script": final_script,
                    "parts": [{"name": p["name"], "stl_url": get_file_url(p["stl_path"])} for p in (exec_result.parts or [])],
                    "bom_suggestion": bom, "warnings": all_warnings, "attempts": 1,
                    "generation_time_s": round(time.time() - start_time, 2), "part_id": part_id,
                    "pipeline": "template" if template_script else "csg_builder",
                    "confidence": decomposed.confidence, "object_name": decomposed.object_name,
                    "dims_estimated": decomposed.dims_are_estimated,
                })
                return
            logger.info(f"Fast path failed ({exec_result.error[:100]}) — falling back to AI")

        # Fine-tuned model fast path
        tuned_model_name = get_active_tuned_model(SessionLocal())
        if tuned_model_name:
            yield sse("progress", {"stage": "generating", "label": "Generating script with fine-tuned model...", "done": False})
            tuned_result = await asyncio.to_thread(generate_with_tuned_model, tuned_model_name, req.description, req.manufacturing_method)
            if tuned_result.success:
                yield sse("progress", {"stage": "executing", "label": "Executing fine-tuned script...", "done": False})
                exec_result = await asyncio.to_thread(execute_cadquery_sandboxed, tuned_result.script)
                if exec_result.success:
                    yield sse("progress", {"stage": "exporting", "label": "Exporting STL...", "done": True})
                    bom = extract_bom_from_script(tuned_result.script)
                    part_id = _save_generated_part(
                        user_id=user_id, description=req.description,
                        manufacturing_method=req.manufacturing_method, script=tuned_result.script,
                        stl_path=exec_result.stl_path, step_path=exec_result.step_path,
                        bom=bom, warnings=all_warnings, attempts=1,
                        generation_time_s=round(time.time() - start_time, 2),
                    )
                    yield sse("result", {
                        "success": True,
                        "stl_url": get_file_url(exec_result.stl_path) if exec_result.stl_path else None,
                        "step_url": get_file_url(exec_result.step_path) if exec_result.step_path else None,
                        "script": tuned_result.script, "bom_suggestion": bom, "warnings": all_warnings,
                        "attempts": 1, "generation_time_s": round(time.time() - start_time, 2), "part_id": part_id,
                        "pipeline": "tuned_model", "confidence": decomposed.confidence,
                        "object_name": decomposed.object_name, "dims_estimated": decomposed.dims_are_estimated,
                    })
                    return
                logger.info("Tuned model script failed execution — falling back to base AI pipeline")

        # Shape research
        yield sse("progress", {"stage": "planning", "label": "Researching object shape...", "done": False})
        shape_research = None
        try:
            shape_research = await asyncio.to_thread(research_shape, req.description)
        except Exception as e:
            logger.warning(f"Shape research failed (non-fatal): {e}")

        # Planning pass
        yield sse("progress", {"stage": "planning", "label": "Designing geometry...", "done": False})
        plan_json = None
        plan_user_msg = f"Part description: {req.description}"
        if req.constraints:
            cl = [f"  - {k}: {v}" for k, v in req.constraints.items()]
            plan_user_msg += "\n\nDimensional constraints:\n" + "\n".join(cl)
        plan_user_msg += f"\n\nManufacturing method: {req.manufacturing_method}"

        try:
            planning_prompt = SHAPE_RESEARCH_PROMPT_TO_PLAN.format(shape_research=shape_research) if shape_research else DESIGN_PLAN_PROMPT
            raw_plan = await asyncio.to_thread(_generate_content, MODEL_FLASH, planning_prompt, plan_user_msg)
            plan_json = parse_json_response(raw_plan)
        except Exception as e:
            logger.warning(f"Planning pass failed: {e}")

        yield sse("progress", {"stage": "planning", "label": "Designing geometry...", "done": True})
        yield sse("progress", {"stage": "generating", "label": "Generating CadQuery script...", "done": False})

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
            if shape_research:
                user_message += "\n\n" + SHAPE_RESEARCH_INJECTION.format(shape_research=shape_research)
        else:
            user_message = f"Generate a CadQuery script for: {req.description}"
            if shape_research:
                user_message += "\n\n" + SHAPE_RESEARCH_INJECTION.format(shape_research=shape_research)
            if req.constraints:
                cl = [f"  - {k}: {v}" for k, v in req.constraints.items()]
                user_message += "\n\nDimensional constraints:\n" + "\n".join(cl)
            user_message += f"\n\nManufacturing method: {req.manufacturing_method}"

        try:
            raw_text = await asyncio.to_thread(_generate_content, MODEL_PRO, system_prompt, user_message)
            script = extract_python_code(raw_text)
        except Exception as e:
            yield sse("result", {"success": False, "error": f"Failed to generate script: {str(e)}"})
            return

        yield sse("progress", {"stage": "generating", "label": "Generating CadQuery script...", "done": True})

        is_valid, validation_warnings = validate_script(script)
        all_warnings.extend(validation_warnings)
        if not is_valid:
            yield sse("result", {"success": False, "error": "Script failed safety validation",
                                  "warnings": all_warnings, "script": script})
            return

        yield sse("progress", {"stage": "executing", "label": "Building 3D model...", "done": False})

        result = await asyncio.to_thread(execute_cadquery_sandboxed, script)
        attempts = 1

        if not result.success:
            current_error = result.error
            for attempt in range(MAX_RETRIES):
                attempt_num = attempt + 2
                yield sse("progress", {"stage": "executing",
                                        "label": f"Fixing errors (attempt {attempt_num})...", "done": False})
                retry_msg = RETRY_PROMPT_TEMPLATE.format(error=current_error, script=script)
                try:
                    raw_retry = await asyncio.to_thread(_generate_content, MODEL_FLASH, system_prompt, retry_msg)
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
            yield sse("result", {"success": False, "error": result.error, "script": script,
                                  "warnings": all_warnings, "attempts": attempts,
                                  "generation_time_s": round(total_time, 2)})
            return

        # AI validation pass
        yield sse("progress", {"stage": "validating", "label": "Validating design accuracy...", "done": False})
        validation_confidence = decomposed.confidence
        try:
            from services.ai_validator import validate_and_correct
            validated_script, validation = await validate_and_correct(script, req.description)
            validation_confidence = validation.confidence
            if validation.corrections_applied > 0 and validated_script != script:
                is_valid_v, _ = validate_script(validated_script)
                if is_valid_v:
                    corrected_result = await asyncio.to_thread(execute_cadquery_sandboxed, validated_script)
                    if corrected_result.success:
                        script = validated_script
                        result = corrected_result
                        all_warnings.append(f"AI validation corrected {validation.corrections_applied} round(s), "
                                            f"confidence: {validation.confidence:.0%}")
                    else:
                        all_warnings.append(f"AI validation confidence: {validation.confidence:.0%} (correction failed)")
                else:
                    all_warnings.append(f"AI validation confidence: {validation.confidence:.0%} (correction failed lint)")
            elif validation.confidence < 0.7:
                all_warnings.append(f"AI validation confidence: {validation.confidence:.0%} — model may not fully match description")
        except Exception as e:
            logger.warning(f"AI validation failed (non-fatal): {e}")

        yield sse("progress", {"stage": "validating", "label": "Validation complete", "done": True})
        yield sse("progress", {"stage": "exporting", "label": "Exporting STL...", "done": True})

        bom = extract_bom_from_script(script)
        part_id = _save_generated_part(
            user_id=user_id, description=req.description, manufacturing_method=req.manufacturing_method,
            script=script, stl_path=result.stl_path, step_path=result.step_path,
            bom=bom, warnings=all_warnings, attempts=attempts, generation_time_s=round(total_time, 2),
        )

        yield sse("result", {
            "success": True,
            "stl_url": get_file_url(result.stl_path) if result.stl_path else None,
            "step_url": get_file_url(result.step_path) if result.step_path else None,
            "script": script,
            "parts": [{"name": p["name"], "stl_url": get_file_url(p["stl_path"])} for p in (result.parts or [])],
            "bom_suggestion": bom, "warnings": all_warnings, "attempts": attempts,
            "generation_time_s": round(total_time, 2), "part_id": part_id,
            "pipeline": "ai_fallback", "confidence": validation_confidence,
            "object_name": decomposed.object_name, "dims_estimated": decomposed.dims_are_estimated,
            "design_notes": plan_json,
        })

    return StreamingResponse(
        event_gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/from-step/stream")
async def generate_from_step_stream(req: FromStepRequest, user=Depends(get_current_user)):
    """Generate or modify a part that references an uploaded STEP file, streaming via SSE."""
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

        yield sse("progress", {"stage": "understanding", "label": "Reading STEP reference...", "done": True})
        yield sse("progress", {"stage": "planning", "label": "Planning modifications...", "done": False})

        system_prompt = STEP_IMPORT_SYSTEM_PROMPT
        user_message = (
            f"Reference STEP file: uploaded_part.step\n\n"
            f"Task: {req.description}\n\n"
            f"Manufacturing method: {req.manufacturing_method}\n\n"
            "Write a CadQuery script that imports the STEP file and performs the task."
        )

        yield sse("progress", {"stage": "planning", "label": "Planning modifications...", "done": True})
        yield sse("progress", {"stage": "generating", "label": "Generating CadQuery script...", "done": False})

        try:
            raw_text = await asyncio.to_thread(_generate_content, MODEL_PRO, system_prompt, user_message)
            script = extract_python_code(raw_text)
        except Exception as e:
            yield sse("result", {"success": False, "error": f"Failed to generate script: {e}"})
            return

        yield sse("progress", {"stage": "generating", "label": "Generating CadQuery script...", "done": True})

        is_valid, validation_warnings = validate_script(script)
        all_warnings.extend(validation_warnings)
        if not is_valid:
            yield sse("result", {"success": False, "error": "Script failed safety validation",
                                  "warnings": all_warnings, "script": script})
            return

        yield sse("progress", {"stage": "executing", "label": "Building 3D model...", "done": False})

        extra_files = {"uploaded_part.step": step_path}
        result = await asyncio.to_thread(execute_cadquery_sandboxed, script, EXECUTION_TIMEOUT, True, extra_files)
        attempts = 1

        if not result.success:
            current_error = result.error
            for attempt in range(MAX_RETRIES):
                attempt_num = attempt + 2
                yield sse("progress", {"stage": "executing",
                                        "label": f"Fixing errors (attempt {attempt_num})...", "done": False})
                retry_msg = RETRY_PROMPT_TEMPLATE.format(error=current_error, script=script)
                try:
                    raw_retry = await asyncio.to_thread(_generate_content, MODEL_FLASH, system_prompt, retry_msg)
                    script = extract_python_code(raw_retry)
                    result = await asyncio.to_thread(execute_cadquery_sandboxed, script, EXECUTION_TIMEOUT, True, extra_files)
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
            user_id=user_id, description=f"[STEP-based] {req.description}",
            manufacturing_method=req.manufacturing_method, script=script,
            stl_path=result.stl_path, step_path=result.step_path,
            bom=bom, warnings=all_warnings, attempts=attempts, generation_time_s=round(total_time, 2),
        )

        yield sse("result", {
            "success": True,
            "stl_url": get_file_url(result.stl_path) if result.stl_path else None,
            "step_url": get_file_url(result.step_path) if result.step_path else None,
            "script": script, "bom_suggestion": bom, "warnings": all_warnings,
            "attempts": attempts, "generation_time_s": round(total_time, 2), "part_id": part_id,
        })

    return StreamingResponse(
        event_gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
