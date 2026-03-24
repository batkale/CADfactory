"""
Layer 1 — NLP Contextual Extraction

Parses a freeform user prompt and separates:
  - Hard physical constraints (explicit measurements, material specs)
  - Situational intent (use-case, ergonomics, aesthetics, context)
  - Object category (functional / decorative / mechanical / organic)

Uses a single fast Gemini Flash call to produce a structured ContextBundle.

No units or numbers are processed here — that is Layer 2's job.

Location: cadfactory-backend/services/nlp_extractor.py
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Output model ───────────────────────────────────────────────────────────────

class ContextBundle(BaseModel):
    """Structured output of Layer 1 — NLP Contextual Extraction."""

    # What the user fundamentally wants to make
    object_category: str  # e.g. "container", "fastener", "bracket", "decorative", "enclosure"

    # High-level intent type
    intent_type: str  # "functional" | "decorative" | "mechanical" | "organic" | "structural"

    # Explicit, hard constraints stated by the user (verbatim value strings, not yet parsed)
    # e.g. {"width": "50mm", "capacity": "300ml", "thread": "M6"}
    raw_constraints: Dict[str, str] = Field(default_factory=dict)

    # Implied/situational context — free-form notes the AI extracted
    # e.g. ["should be ergonomic", "for bathroom shelf", "needs to withstand heat"]
    situational_notes: List[str] = Field(default_factory=list)

    # Suggested manufacturing process (if inferable)
    manufacturing_hint: Optional[str] = None  # "fdm" | "sla" | "cnc" | "injection" | None

    # Material hint (if mentioned or inferable)
    material_hint: Optional[str] = None  # "ABS" | "PLA" | "aluminum" | None

    # Detected modifiers that affect geometry style
    # e.g. ["rounded", "slim", "ribbed", "hollow", "symmetric"]
    geometric_modifiers: List[str] = Field(default_factory=list)

    # Whether this is a multi-part assembly request (triggers complexity warning)
    is_assembly: bool = False

    # Whether the user asked for organic / non-geometric shapes
    is_organic: bool = False

    # Confidence that the extraction is complete and unambiguous
    confidence: float = 0.0

    # If low confidence, a clarification question
    clarification_question: Optional[str] = None


# ── Layer 1 system prompt ──────────────────────────────────────────────────────

_LAYER1_SYSTEM_PROMPT = """
You are Layer 1 of a precision 3D CAD pipeline: the NLP Contextual Extractor.

Your sole task is to parse the user's description and output a structured JSON that
classifies WHAT they want and separates:

  (A) HARD CONSTRAINTS — explicit physical values and specifications the user stated.
      Examples: "50mm wide", "M6 thread", "300ml capacity", "2mm wall", "triangular"
      → Output as raw_constraints dict with descriptive keys and verbatim value strings.

  (B) SITUATIONAL CONTEXT — implied needs, use-cases, and qualitative preferences.
      Examples: "for a bathroom shelf", "needs to be sturdy", "ergonomic grip", "light"
      → Output as situational_notes list of concise factual statements.

  (C) OBJECT CLASSIFICATION:
      object_category: one of: container, fastener, bracket, enclosure, decorative,
                                mount, connector, tool_part, organic, furniture_part, other
      intent_type:     one of: functional, decorative, mechanical, organic, structural

  (D) PROCESS & MATERIAL HINTS — only if clearly stated or strongly implied.
      manufacturing_hint: fdm | sla | cnc | injection | casting | null
      material_hint: ABS | PLA | PETG | aluminum | steel | wood | null

  (E) GEOMETRIC MODIFIERS — adjectives that alter shape (rounded, hollow, symmetric,
      ribbed, tapered, perforated, latticed, chamfered, slit, etc.)

  (F) is_assembly: true ONLY if user asks for multiple distinct parts that mate together.
  (G) is_organic: true if the shape is freeform, non-prismatic, or biologically inspired.

RULES:
- Do NOT parse numbers or convert units — just copy value strings verbatim.
- Do NOT hallucinate constraints not present in the input.
- If the prompt is very short / ambiguous, set confidence < 0.7 and provide
  clarification_question.
- Keep situational_notes to at most 5 concise sentences.
- Return ONLY valid JSON, no markdown, no explanation.

JSON format:
{
  "object_category": "...",
  "intent_type": "...",
  "raw_constraints": {"key": "value_string", ...},
  "situational_notes": ["...", ...],
  "manufacturing_hint": "fdm" | null,
  "material_hint": "ABS" | null,
  "geometric_modifiers": ["...", ...],
  "is_assembly": false,
  "is_organic": false,
  "confidence": 0.85,
  "clarification_question": null
}
"""


# ── Core function ──────────────────────────────────────────────────────────────

def extract_context(prompt: str) -> ContextBundle:
    """
    Layer 1: Parse a freeform prompt into a structured ContextBundle.

    Uses Gemini Flash for speed. Always returns a ContextBundle — never raises.
    Falls back to a minimal bundle on API failure.
    """
    from services.claude_cad import _generate_content, MODEL_FLASH
    from services.script_utils import parse_json_response

    data = None
    for attempt in range(2):
        try:
            raw = _generate_content(MODEL_FLASH, _LAYER1_SYSTEM_PROMPT, prompt)
            data = parse_json_response(raw)
        except Exception as exc:
            logger.warning(f"Layer 1 API call failed (attempt {attempt + 1}): {exc}")
            break

        if data and isinstance(data, dict):
            break

        logger.warning(f"Layer 1: non-dict response (attempt {attempt + 1}), retrying…")
        data = None

    if not data:
        return _fallback_bundle(prompt)

    return ContextBundle(
        object_category=str(data.get("object_category", "other")),
        intent_type=str(data.get("intent_type", "functional")),
        raw_constraints=dict(data.get("raw_constraints") or {}),
        situational_notes=list(data.get("situational_notes") or []),
        manufacturing_hint=data.get("manufacturing_hint"),
        material_hint=data.get("material_hint"),
        geometric_modifiers=list(data.get("geometric_modifiers") or []),
        is_assembly=bool(data.get("is_assembly", False)),
        is_organic=bool(data.get("is_organic", False)),
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
        clarification_question=data.get("clarification_question"),
    )


def _fallback_bundle(prompt: str) -> ContextBundle:
    """Minimal safe fallback when Layer 1 extraction fails."""
    return ContextBundle(
        object_category="other",
        intent_type="functional",
        raw_constraints={},
        situational_notes=[f"Raw prompt: {prompt[:120]}"],
        confidence=0.3,
        clarification_question="Could you describe the shape, size, and purpose more specifically?",
    )
