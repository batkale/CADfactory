"""
Refine, execute, decompose, and suggest-loads endpoints for /api/generate.
"""
import asyncio
import json as _json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

import security as auth_utils
from config.prompts import REFINE_SYSTEM_PROMPT, RETRY_PROMPT_TEMPLATE, build_system_prompt
from services.cadquery_runner import EXECUTION_TIMEOUT, execute_cadquery_sandboxed
from services.claude_cad import MAX_RETRIES, MODEL_FLASH, _generate_content, refine_script, suggest_load_cases
from services.script_utils import extract_bom_from_script, extract_python_code, validate_script
from services.semantic_decomposer import decompose_prompt
from routers.generate_schemas import (
    DecomposeRequest,
    ExecuteRequest,
    GenerateResponse,
    RefineRequest,
    SuggestLoadsRequest,
    _save_generated_part,
)
from services.cadquery_runner import get_file_url

get_current_user = auth_utils.get_current_user
logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


@router.post("/refine", response_model=GenerateResponse)
async def refine_part(req: RefineRequest, user=Depends(get_current_user)):
    """Refine an existing generated part via natural language."""
    logger.info(f"Refine request from {getattr(user, 'email', '?')}: {req.message[:60]}...")

    result = await refine_script(
        current_script=req.current_script,
        user_message=req.message,
        conversation_history=req.conversation_history,
        manufacturing_method=req.manufacturing_method,
    )

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail={"error": result.error, "script": result.script, "warnings": result.warnings},
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
    """Streaming version of /refine via SSE."""
    user_id = user.id

    async def event_gen():
        import time
        start_time = time.time()

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

        all_warnings: list = []

        yield sse("progress", {"stage": "understanding", "label": "Analysing modification request...", "done": False})
        yield sse("progress", {"stage": "understanding", "label": "Analysing modification request...", "done": True})
        yield sse("progress", {"stage": "generating", "label": "Applying changes to script...", "done": False})

        refine_msg = (
            f"Current script:\n```python\n{req.current_script}\n```\n\n"
            f"Modification request: {req.message}"
        )
        try:
            raw_text = await asyncio.to_thread(_generate_content, MODEL_FLASH, REFINE_SYSTEM_PROMPT, refine_msg)
            script = extract_python_code(raw_text)
        except Exception as e:
            yield sse("result", {"success": False, "error": f"Failed to modify script: {e}"})
            return

        yield sse("progress", {"stage": "generating", "label": "Applying changes to script...", "done": True})

        is_valid, validation_warnings = validate_script(script)
        all_warnings.extend(validation_warnings)
        if not is_valid:
            yield sse("result", {"success": False, "error": "Script failed safety validation",
                                  "script": script, "warnings": all_warnings})
            return

        yield sse("progress", {"stage": "executing", "label": "Building 3D model...", "done": False})

        result = await asyncio.to_thread(execute_cadquery_sandboxed, script)
        attempts = 1

        if not result.success:
            system_prompt = build_system_prompt(req.manufacturing_method)
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

        yield sse("progress", {"stage": "exporting", "label": "Exporting STL...", "done": True})

        bom = extract_bom_from_script(script)
        part_id = _save_generated_part(
            user_id=user_id, description=req.message, manufacturing_method=req.manufacturing_method,
            script=script, stl_path=result.stl_path, step_path=result.step_path,
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


@router.post("/execute", response_model=GenerateResponse)
async def execute_script(req: ExecuteRequest, user=Depends(get_current_user)):
    """Execute a CadQuery script directly without AI generation."""
    is_valid, validation_warnings = validate_script(req.script)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={"error": "Script failed safety validation", "warnings": validation_warnings},
        )

    result = await asyncio.to_thread(execute_cadquery_sandboxed, req.script)
    if not result.success:
        raise HTTPException(
            status_code=422,
            detail={"error": result.error, "script": req.script, "warnings": validation_warnings},
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
    """Get AI-suggested load cases for topology optimization."""
    suggestions = await suggest_load_cases(
        description=req.description,
        bbox=(req.bbox_x, req.bbox_y, req.bbox_z),
        volume=req.volume,
        surface_area=req.surface_area,
    )
    if not suggestions:
        raise HTTPException(status_code=500, detail="Failed to generate load case suggestions")
    return suggestions


@router.post("/decompose")
async def decompose_part_description(req: DecomposeRequest, user=Depends(get_current_user)):
    """Run semantic decomposition on a prompt and return the CSG tree (debug/preview)."""
    result = await asyncio.to_thread(decompose_prompt, req.prompt)
    return result.model_dump()
