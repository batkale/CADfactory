"""
Image-to-CAD endpoints for /api/generate.
"""
import asyncio
import json as _json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

import security as auth_utils
from config.prompts import DESIGN_PLAN_PROMPT, DESIGN_PLAN_TO_CODE_TEMPLATE
from services.cadquery_runner import execute_cadquery_sandboxed, get_file_url
from services.claude_cad import MAX_RETRIES, MODEL_FLASH, MODEL_PRO, _generate_content
from services.script_utils import extract_bom_from_script, extract_python_code, parse_json_response, validate_script
from routers.generate_schemas import (
    ImageToCADResponse,
    _save_generated_part,
)

get_current_user = auth_utils.get_current_user
logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


def _validate_image_upload(file: UploadFile) -> None:
    """Raise HTTPException if the uploaded file is not an accepted image type."""
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. Allowed: {', '.join(allowed_types)}",
        )


@router.post("/from-image", response_model=ImageToCADResponse)
async def generate_from_image(
    request: Request,
    file: UploadFile = File(None),
    additional_context: str = Form(""),
    manufacturing_method: str = Form("fdm"),
    current_user=Depends(get_current_user),
):
    """
    Analyze an image and extract a structured CAD plan.
    Flow: Image → Edge preprocessing → Gemini Vision → Structured plan + description.
    """
    from services.image_to_cad import image_to_cad_plan

    if file is None:
        raise HTTPException(status_code=400, detail="No image file uploaded")

    _validate_image_upload(file)

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    try:
        result = await image_to_cad_plan(
            image_bytes=image_bytes,
            filename=file.filename or "image.png",
            additional_context=additional_context,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Image-to-CAD analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Image analysis failed")

    return ImageToCADResponse(
        extracted_description=result.description,
        structured_plan=result.structured_plan,
        confidence=result.confidence,
        message="Image analyzed. Use the description to generate your CAD model.",
    )


@router.post("/from-image/stream")
async def generate_from_image_stream(
    request: Request,
    file: UploadFile = File(None),
    additional_context: str = Form(""),
    manufacturing_method: str = Form("fdm"),
    current_user=Depends(get_current_user),
):
    """Full image → CAD pipeline with SSE streaming."""
    from services.image_to_cad import image_to_cad_plan

    if file is None:
        raise HTTPException(status_code=400, detail="No image file uploaded")

    _validate_image_upload(file)

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    _filename = file.filename or "image.png"
    _additional_context = additional_context
    _manufacturing_method = manufacturing_method
    _user_id = current_user.id

    async def event_gen():
        import time

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

        yield sse("progress", {"stage": "analyzing_image", "label": "Analyzing image...", "done": False})

        try:
            img_result = await image_to_cad_plan(
                image_bytes=image_bytes,
                filename=_filename,
                additional_context=_additional_context,
            )
        except Exception as e:
            yield sse("result", {"success": False, "error": f"Image analysis failed: {e}"})
            return

        yield sse("progress", {"stage": "analyzing_image", "label": "Image analyzed", "done": True})
        yield sse("image_analysis", {
            "description": img_result.description,
            "structured_plan": img_result.structured_plan,
            "confidence": img_result.confidence,
        })

        description = img_result.description
        if not description:
            yield sse("result", {"success": False, "error": "Could not extract a description from the image."})
            return

        _injected_plan = img_result.structured_plan

        start_time = time.time()
        all_warnings: list = []

        yield sse("progress", {"stage": "understanding", "label": "Understanding description...", "done": False})

        rag_examples: list = []
        try:
            from services.rag_store import RAGStore
            rag_store = RAGStore()
            rag_examples = await asyncio.to_thread(rag_store.retrieve, description, k=3)
        except Exception as e:
            logger.warning(f"RAG retrieval failed (non-fatal): {e}")

        try:
            from services.semantic_decomposer import SemanticDecomposer
            decomposer = SemanticDecomposer()
            decomposed = await asyncio.to_thread(decomposer.decompose, description)
        except Exception as e:
            logger.warning(f"Semantic decomposer failed (non-fatal): {e}")
            from types import SimpleNamespace
            decomposed = SimpleNamespace(confidence=0.7, object_name=None, dims_are_estimated=True)

        yield sse("progress", {"stage": "understanding", "label": "Understanding description...", "done": True})
        yield sse("progress", {"stage": "planning", "label": "Designing geometry...", "done": False})

        plan_json = _injected_plan
        if plan_json is None:
            try:
                plan_user_msg = f"Part description: {description}\nManufacturing method: {_manufacturing_method}"
                raw_plan = await asyncio.to_thread(_generate_content, MODEL_FLASH, DESIGN_PLAN_PROMPT, plan_user_msg)
                plan_json = parse_json_response(raw_plan)
            except Exception as e:
                logger.warning(f"Planning pass failed: {e}")

        yield sse("progress", {"stage": "planning", "label": "Designing geometry...", "done": True})
        yield sse("progress", {"stage": "generating", "label": "Generating CadQuery script...", "done": False})

        from services.claude_cad import build_system_prompt_with_examples
        system_prompt = build_system_prompt_with_examples(_manufacturing_method, rag_examples)

        if plan_json:
            user_message = DESIGN_PLAN_TO_CODE_TEMPLATE.format(
                plan=_json.dumps(plan_json, indent=2),
                manufacturing_method=_manufacturing_method,
                constraints="",
            )
        else:
            user_message = f"Generate a CadQuery script for: {description}\nManufacturing method: {_manufacturing_method}"

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

        result = await asyncio.to_thread(execute_cadquery_sandboxed, script)
        attempts = 1

        if not result.success:
            current_error = result.error
            for attempt in range(MAX_RETRIES):
                attempt_num = attempt + 2
                yield sse("progress", {"stage": "executing",
                                        "label": f"Fixing errors (attempt {attempt_num})...", "done": False})
                from config.prompts import RETRY_PROMPT_TEMPLATE
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

        yield sse("progress", {"stage": "executing", "label": "Build complete", "done": True})
        yield sse("progress", {"stage": "exporting", "label": "Exporting STL...", "done": True})

        bom = extract_bom_from_script(script)
        part_id = _save_generated_part(
            user_id=_user_id, description=description, manufacturing_method=_manufacturing_method,
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
            "pipeline": "image_to_cad", "design_notes": plan_json,
        })

    return StreamingResponse(
        event_gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
