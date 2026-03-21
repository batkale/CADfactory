"""
Gemini API integration for CadQuery code generation.
Handles prompt building, API calls, and the self-healing retry loop
when generated scripts fail execution.

Location: cadfactory-backend/services/claude_cad.py
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types

from config.prompts import (
    build_system_prompt, build_system_prompt_with_examples,
    RETRY_PROMPT_TEMPLATE, LOAD_SUGGESTION_PROMPT,
    REFINE_SYSTEM_PROMPT, DESIGN_PLAN_PROMPT, DESIGN_PLAN_TO_CODE_TEMPLATE,
)
from services.script_utils import extract_python_code, validate_script, parse_json_response
from services.cadquery_runner import execute_cadquery_sandboxed, ExecutionResult

logger = logging.getLogger(__name__)

# Gemini API configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_PRO = os.environ.get("GEMINI_MODEL_PRO", "gemini-2.5-pro")      # Smarter, for initial generation
MODEL_FLASH = os.environ.get("GEMINI_MODEL_FLASH", "gemini-2.5-flash") # Faster, for retries
MAX_RETRIES = int(os.environ.get("CADQUERY_MAX_RETRIES", "3"))

# Configure the SDK client
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Get your key at https://aistudio.google.com/app/apikey"
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


@dataclass
class GenerationResult:
    """Full result from the generate-then-execute pipeline."""
    success: bool
    script: str = ""
    stl_path: Optional[str] = None
    step_path: Optional[str] = None
    bom_suggestion: list = None
    warnings: list = None
    error: Optional[str] = None
    attempts: int = 1
    total_time_s: float = 0.0

    def __post_init__(self):
        if self.bom_suggestion is None:
            self.bom_suggestion = []
        if self.warnings is None:
            self.warnings = []


def _generate_content(model_name: str, system_prompt: str, user_message: str) -> str:
    """Single-turn generation with system prompt."""
    client = _get_client()
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config=config,
    )
    return response.text


def _chat_send(chat, message: str) -> str:
    """Send a message in a chat session and return the response text."""
    response = chat.send_message(message)
    return response.text


def _start_chat(model_name: str, system_prompt: str, history: list):
    """Start a chat session with prior history."""
    client = _get_client()
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
    )
    return client.chats.create(
        model=model_name,
        config=config,
        history=history,
    )


def get_model(system_prompt: str = "", use_pro: bool = True):
    """Return (model_name, system_prompt) — kept for backwards compat."""
    model_name = MODEL_PRO if use_pro else MODEL_FLASH
    return model_name, system_prompt


def generate_with_tuned_model(
    tuned_model_name: str,
    description: str,
    manufacturing_method: str = "fdm",
) -> GenerationResult:
    """
    Single-pass CadQuery script generation using a fine-tuned model.

    The tuned model has been trained on (description, script) pairs from
    thumbs-up feedback, so it can generate domain-appropriate scripts in one
    call — no planning pass needed.

    Returns GenerationResult with success=False if generation or extraction fails.
    The caller should fall back to the standard two-pass pipeline on failure.
    """
    from config.prompts import build_system_prompt
    from services.script_utils import extract_python_code

    system_prompt = build_system_prompt(manufacturing_method)
    user_message = (
        f"Generate a CadQuery Python script for the following 3D part.\n"
        f"Description: {description}\n"
        f"Manufacturing method: {manufacturing_method}\n\n"
        f"Return only executable Python code. Use show_object(result) at the end."
    )
    try:
        raw = _generate_content(tuned_model_name, system_prompt, user_message)
        script = extract_python_code(raw)
        if not script:
            return GenerationResult(
                success=False,
                error="Tuned model returned no extractable Python code",
            )
        return GenerationResult(success=True, script=script, attempts=1)
    except Exception as e:
        logger.warning(f"Tuned model generation failed: {e}")
        return GenerationResult(success=False, error=str(e))


def check_complexity(description: str) -> Optional[str]:
    """
    Check if a part description is too complex for reliable CadQuery generation.
    Returns an error message if too complex, None if OK.
    """
    desc_lower = description.lower()

    # Multi-body / assembly keywords — each tuple counts as ONE complexity point
    assembly_keywords = [
        ("gear", "gears", "involute gear", "spur gear"),
        ("timer", "clock", "mechanism"),
        ("linkage", "hinge joint"),
        ("coil spring",),
        ("involute", "tooth profile"),
        ("cam", "camshaft"),
        ("bearing assembly",),
        ("chain drive", "sprocket chain"),
        ("planetary gear",),
    ]

    for keywords in assembly_keywords:
        for kw in keywords:
            if kw in desc_lower:
                # Count ONE point per matching tuple (not per keyword) to avoid substring false positives
                complex_count = sum(
                    1 for kws in assembly_keywords
                    if any(k in desc_lower for k in kws)
                )
                if complex_count >= 2:
                    return (
                        f"This design is too complex for single-step generation. "
                        f"AI generation works best with single-body parts like brackets, "
                        f"enclosures, plates, and mounts.\n\n"
                        f"Try breaking it into separate parts:\n"
                        f"  • Generate each component individually\n"
                        f"  • 'motor mount with 25mm bore and M3 holes'\n"
                        f"  • 'spur gear 20 teeth, module 1.5, 8mm bore'\n"
                        f"  • 'gear housing 80x60x30mm with bearing seats'\n\n"
                        f"Then assemble them in your CAD software or slicer."
                    )

    # Check for multiple distinct parts requested
    multi_part_signals = ["and a ", " with 2 ", " with 3 ", " two ", " three ",
                          " both ", " assembly of ", " complete ", " full "]
    multi_count = sum(1 for s in multi_part_signals if s in desc_lower)

    part_count_words = [" gears", " motors", " shafts", " bearings", " springs",
                        " pulleys", " wheels", " arms", " links"]
    plural_parts = sum(1 for p in part_count_words if p in desc_lower)

    if multi_count >= 2 or plural_parts >= 2:
        return (
            f"This looks like a multi-part assembly. AI generation works best "
            f"with one part at a time.\n\n"
            f"Try generating each component separately, then combine them "
            f"in your analysis."
        )

    return None  # Complexity is OK


async def generate_script(
    description: str,
    manufacturing_method: str = "fdm",
    constraints: Optional[dict] = None,
) -> str:
    """
    Two-pass generation:
      Pass 1 — Flash produces a structured JSON design plan (no code).
      Pass 2 — Pro takes that plan + system prompt examples and writes CadQuery code.
    Falls back to single-pass if planning fails.
    """
    import json
    from services import rag_store

    # ── RAG: fetch similar approved examples ──────────────────────────────────
    rag_examples = []
    try:
        rag_examples = rag_store.get_similar_examples(description, manufacturing_method, k=3)
        if rag_examples:
            logger.info(f"RAG: injecting {len(rag_examples)} similar example(s) into prompt")
    except Exception as e:
        logger.warning(f"RAG lookup failed (non-fatal): {e}")

    system_prompt = build_system_prompt_with_examples(manufacturing_method, rag_examples)

    # ── Pass 1: design planning (Flash, cheap + fast) ─────────────────────────
    plan_user_msg = f"Part description: {description}"
    if constraints:
        constraint_lines = [f"  - {k}: {v}" for k, v in constraints.items()]
        plan_user_msg += "\n\nDimensional constraints:\n" + "\n".join(constraint_lines)
    plan_user_msg += f"\n\nManufacturing method: {manufacturing_method}"

    logger.info(f"Pass 1 — design planning: {description[:80]}...")
    plan_json = None
    try:
        raw_plan = _generate_content(MODEL_FLASH, DESIGN_PLAN_PROMPT, plan_user_msg)
        plan_json = parse_json_response(raw_plan)
        if plan_json:
            logger.info(f"Pass 1 plan: {json.dumps(plan_json)[:200]}")
        else:
            logger.warning("Pass 1 returned no valid JSON — falling back to single-pass")
    except Exception as e:
        logger.warning(f"Pass 1 failed ({e}) — falling back to single-pass")

    # ── Pass 2: code generation (Pro, accurate) ───────────────────────────────
    if plan_json:
        constraints_str = ""
        if constraints:
            constraint_lines = [f"  - {k}: {v}" for k, v in constraints.items()]
            constraints_str = "\n\nDimensional constraints:\n" + "\n".join(constraint_lines)

        user_message = DESIGN_PLAN_TO_CODE_TEMPLATE.format(
            plan=json.dumps(plan_json, indent=2),
            manufacturing_method=manufacturing_method,
            constraints=constraints_str,
        )
        logger.info("Pass 2 — code generation from plan...")
    else:
        # Single-pass fallback
        user_message = f"Generate a CadQuery script for: {description}"
        if constraints:
            constraint_lines = [f"  - {k}: {v}" for k, v in constraints.items()]
            user_message += "\n\nDimensional constraints:\n" + "\n".join(constraint_lines)
        user_message += f"\n\nManufacturing method: {manufacturing_method}"
        logger.info("Single-pass generation (fallback)...")

    raw_text = _generate_content(MODEL_PRO, system_prompt, user_message)
    script = extract_python_code(raw_text)

    logger.info(f"Generated script: {len(script)} chars, {script.count(chr(10))+1} lines")
    return script


async def generate_and_execute(
    description: str,
    manufacturing_method: str = "fdm",
    constraints: Optional[dict] = None,
) -> GenerationResult:
    """
    Full pipeline: generate CadQuery script -> execute -> retry on failure.
    This is the main entry point called by the router.
    """
    import time
    from services.script_utils import extract_bom_from_script

    start_time = time.time()
    all_warnings = []

    # Step 0: Complexity guard — reject prompts that are too complex for single-body CadQuery
    complexity_issue = check_complexity(description)
    if complexity_issue:
        return GenerationResult(
            success=False,
            error=complexity_issue
        )

    # Step 1: Generate initial script
    try:
        script = await generate_script(description, manufacturing_method, constraints)
    except Exception as e:
        return GenerationResult(
            success=False,
            error=f"Failed to generate script: {str(e)}"
        )

    # Step 2: Validate before execution
    is_valid, validation_warnings = validate_script(script)
    all_warnings.extend(validation_warnings)

    if not is_valid:
        return GenerationResult(
            success=False,
            script=script,
            error="Script failed safety validation",
            warnings=all_warnings
        )

    # Step 3: Execute
    result = execute_cadquery_sandboxed(script)

    # Step 4: If it failed, enter the self-healing retry loop
    if not result.success:
        logger.info(f"Initial execution failed, entering retry loop (max {MAX_RETRIES})")
        script, result, attempts = await _retry_with_errors(
            description=description,
            manufacturing_method=manufacturing_method,
            constraints=constraints,
            script=script,
            error=result.error,
        )
        all_warnings.append(f"Script required {attempts} attempt(s) to succeed")
    else:
        attempts = 1

    total_time = time.time() - start_time

    if not result.success:
        return GenerationResult(
            success=False,
            script=script,
            error=result.error,
            warnings=all_warnings,
            attempts=attempts,
            total_time_s=total_time
        )

    # Step 5: Extract BOM suggestions from the working script
    bom = extract_bom_from_script(script)

    return GenerationResult(
        success=True,
        script=script,
        stl_path=result.stl_path,
        step_path=result.step_path,
        bom_suggestion=bom,
        warnings=all_warnings,
        attempts=attempts,
        total_time_s=total_time
    )


async def _retry_with_errors(
    description: str,
    manufacturing_method: str,
    constraints: Optional[dict],
    script: str,
    error: str,
) -> tuple[str, ExecutionResult, int]:
    """
    Self-healing loop: feed CadQuery errors back to Gemini for correction.
    Uses a chat session to maintain conversation context. Uses Flash for speed.
    """
    system_prompt = build_system_prompt(manufacturing_method)
    model_name, _ = get_model(system_prompt, use_pro=False)  # Flash for retries

    # Build the original user message
    original_user_msg = f"Generate a CadQuery script for: {description}"
    if constraints:
        constraint_lines = [f"  - {k}: {v}" for k, v in constraints.items()]
        original_user_msg += "\n\nDimensional constraints:\n" + "\n".join(constraint_lines)
    original_user_msg += f"\n\nManufacturing method: {manufacturing_method}"

    # Start a chat with the original exchange as history
    history = [
        genai_types.Content(role="user", parts=[genai_types.Part(text=original_user_msg)]),
        genai_types.Content(role="model", parts=[genai_types.Part(text=script)]),
    ]
    chat = _start_chat(model_name, system_prompt, history)

    current_script = script
    current_error = error

    for attempt in range(MAX_RETRIES):
        attempt_num = attempt + 2  # +2 because attempt 1 was the initial try

        # Send error feedback
        retry_msg = RETRY_PROMPT_TEMPLATE.format(
            error=current_error,
            script=current_script
        )

        logger.info(f"Retry attempt {attempt_num}/{MAX_RETRIES + 1}: sending error to Gemini")

        try:
            raw_text = _chat_send(chat, retry_msg)
            current_script = extract_python_code(raw_text)
        except Exception as e:
            logger.error(f"Gemini API error on retry {attempt_num}: {e}")
            continue

        # Try executing the fixed script
        result = execute_cadquery_sandboxed(current_script)

        if result.success:
            logger.info(f"Script fixed on attempt {attempt_num}")
            return current_script, result, attempt_num

        current_error = result.error
        logger.warning(f"Retry {attempt_num} still failed: {current_error[:100]}")

    # All retries exhausted
    final_result = ExecutionResult(
        success=False,
        error=f"Failed after {MAX_RETRIES + 1} attempts. Last error: {current_error}"
    )
    return current_script, final_result, MAX_RETRIES + 1


async def refine_script(
    current_script: str,
    user_message: str,
    conversation_history: Optional[list] = None,
    manufacturing_method: str = "fdm",
) -> GenerationResult:
    """
    Conversational refinement: modify an existing script based on user feedback.
    """
    import time
    from services.script_utils import extract_bom_from_script

    start_time = time.time()
    model_name, _ = get_model(use_pro=True)

    refine_msg = (
        f"Current script:\n{current_script}\n\n"
        f"Modification request: {user_message}"
    )

    try:
        raw_text = _generate_content(model_name, REFINE_SYSTEM_PROMPT, refine_msg)
        new_script = extract_python_code(raw_text)
    except Exception as e:
        return GenerationResult(
            success=False,
            script=current_script,
            error=f"Failed to refine script: {str(e)}"
        )

    # Execute the refined script
    result = execute_cadquery_sandboxed(new_script)

    if not result.success:
        # One retry attempt for refinements
        new_script, result, attempts = await _retry_with_errors(
            description=user_message,
            manufacturing_method=manufacturing_method,
            constraints=None,
            script=new_script,
            error=result.error
        )

    total_time = time.time() - start_time
    bom = extract_bom_from_script(new_script) if result.success else []

    return GenerationResult(
        success=result.success,
        script=new_script,
        stl_path=result.stl_path,
        step_path=result.step_path,
        bom_suggestion=bom,
        error=result.error if not result.success else None,
        total_time_s=total_time
    )


async def suggest_load_cases(
    description: str,
    bbox: tuple[float, float, float],
    volume: float,
    surface_area: float,
) -> Optional[dict]:
    """
    Use Gemini to suggest appropriate load cases for topology optimization.
    """
    prompt = LOAD_SUGGESTION_PROMPT.format(
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_z=bbox[2],
        volume=volume,
        volume_cm3=round(volume / 1000, 1),
        surface_area=surface_area,
        description=description
    )

    try:
        model_name, _ = get_model(use_pro=False)  # Flash for simple JSON task
        raw_text = _generate_content(model_name, "", prompt)
        return parse_json_response(raw_text)
    except Exception as e:
        logger.error(f"Failed to suggest load cases: {e}")
        return None