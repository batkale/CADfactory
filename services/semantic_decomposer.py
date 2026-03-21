"""
Semantic Decomposer — converts freeform user prompts into structured CSG trees.

Uses Gemini to identify the object type, real-world reference, symmetry,
estimated dimensions, and a list of CSG operations (add/subtract/intersect)
over geometric primitives.

Location: cadfactory-backend/services/semantic_decomposer.py
"""

from __future__ import annotations

import logging
import math
from typing import Optional, List, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Pydantic models ────────────────────────────────────────────────────────────

class CSGOperation(BaseModel):
    op: Literal["add", "subtract", "intersect"]
    shape: str  # cylinder, box, sphere, cone, torus, rounded_box, polygon_prism
    params: dict  # shape-specific params in mm
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: Optional[List[float]] = None  # [rx, ry, rz] degrees
    copies: Optional[int] = None
    copy_angle_step: Optional[float] = None  # degrees between copies


class DecomposedObject(BaseModel):
    object_name: str
    real_world_reference: str
    symmetry: Optional[str] = None
    estimated_dimensions: dict = Field(default_factory=dict)
    dims_are_estimated: bool = True
    operations: List[CSGOperation] = Field(default_factory=list)
    modifiers: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    clarification_needed: bool = False
    clarification_question: Optional[str] = None


# ── System prompt ──────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM_PROMPT = """
You are a mechanical engineering expert and object analyst for a CAD system.
Given a freeform text description, convert it into a structured CSG (Constructive
Solid Geometry) tree made of geometric primitives.

STEP 1 — OBJECT RECOGNITION:
Identify what the user wants. Map informal or colloquial names to engineering terms:
  "stress wheel" → fidget spinner
  "makeup sponge" → teardrop/egg-shaped foam applicator
  "screw" → bolt/fastener
  "clamp" → C-clamp or bar clamp

STEP 2 — MODIFIER ANALYSIS:
If the user used adjectives or qualifiers (triangular, oval, thin, curved, etc.),
determine WHICH part of the object they modify. Assume the modifier applies to the
most distinctive feature. If multiple interpretations exist, lower the confidence score.

STEP 3 — DIMENSION ESTIMATION:
If the user did not specify dimensions, estimate typical real-world sizes using
common 3D-print scales for hand-held objects. Set dims_are_estimated=true.

STEP 4 — CSG DECOMPOSITION:
Express the object using only these primitives:
  cylinder, box, sphere, cone, torus, rounded_box, polygon_prism, curve_prism

For each primitive, specify:
  - op: "add" | "subtract" | "intersect"
  - shape: primitive name
  - params: shape-specific dimensions in mm (e.g. {"radius": 10, "height": 20})
  - position: [x, y, z] mm from origin
  - rotation: [rx, ry, rz] degrees (optional)
  - copies: integer count for symmetric repetitions (optional)
  - copy_angle_step: degrees between each copy for circular symmetry (optional)

CRITICAL — USE curve_prism FOR ORGANIC / NON-CONVEX SHAPES:
When the object has an organic 2D profile that cannot be expressed as a circle,
square, or regular polygon, use curve_prism instead of approximating with spheres and boxes.
Never approximate hearts with two spheres. Never approximate teardrops with a sphere + cone.
  curve_prism + "heart"    → valentine heart
  curve_prism + "star"     → N-pointed star (n_points, outer_r, inner_r)
  curve_prism + "teardrop" → water drop / teardrop  (width, depth)
  curve_prism + "leaf"     → leaf / lens / petal shape (width, depth)
  curve_prism + "diamond"  → elongated 4-point diamond (width, depth)
  curve_prism + "arrow"    → arrowhead pointing up (width, depth, shaft_width)
  curve_prism + "cross"    → plus / cross shape (width, depth, arm_width)

Cylinder params: {"radius": float, "height": float}
Box params: {"length": float, "width": float, "height": float}
Sphere params: {"radius": float}
Cone params: {"r_base": float, "r_top": float, "height": float}
Torus params: {"major_r": float, "minor_r": float}
Rounded_box params: {"length": float, "width": float, "height": float, "fillet_r": float}
Polygon_prism params: {"sides": int, "diameter": float, "height": float}
Curve_prism params:
  heart    → {"curve": "heart", "width": float, "height": float}
  star     → {"curve": "star", "n_points": int, "outer_r": float, "inner_r": float, "height": float}
  teardrop → {"curve": "teardrop", "width": float, "depth": float, "height": float}
  leaf     → {"curve": "leaf", "width": float, "depth": float, "height": float}
  diamond  → {"curve": "diamond", "width": float, "depth": float, "height": float}
  arrow    → {"curve": "arrow", "width": float, "depth": float, "shaft_width": float, "height": float}
  cross    → {"curve": "cross", "width": float, "depth": float, "arm_width": float, "height": float}

STEP 5 — CONFIDENCE:
Rate your interpretation confidence 0.0–1.0. If below 0.8, set clarification_needed=true
and write a single clarifying question in clarification_question.

Return ONLY valid JSON, nothing else:
{
  "object_name": "short descriptive name",
  "real_world_reference": "common name of the real object",
  "symmetry": "3-fold rotational | bilateral | none | null",
  "estimated_dimensions": {"key": value_mm},
  "dims_are_estimated": true,
  "operations": [
    {
      "op": "add",
      "shape": "cylinder",
      "params": {"radius": 10, "height": 5},
      "position": [0, 0, 0],
      "rotation": null,
      "copies": null,
      "copy_angle_step": null
    }
  ],
  "modifiers": ["list", "of", "shape", "modifiers"],
  "confidence": 0.85,
  "clarification_needed": false,
  "clarification_question": null
}
"""


# ── Core function ──────────────────────────────────────────────────────────────

def decompose_prompt(user_prompt: str) -> DecomposedObject:
    """
    Convert a freeform text prompt into a structured DecomposedObject.

    Calls Gemini Flash (fast, cheap) to analyse the prompt and return a
    CSG tree. Always returns a DecomposedObject — never raises.
    """
    from services.claude_cad import _generate_content, MODEL_FLASH
    from services.script_utils import parse_json_response

    data = None
    for attempt in range(2):  # one retry on malformed response
        try:
            raw = _generate_content(MODEL_FLASH, _DECOMPOSE_SYSTEM_PROMPT, user_prompt)
            data = parse_json_response(raw)
        except Exception as e:
            logger.warning(f"Decompose API call failed (attempt {attempt+1}): {e}")
            break

        if data and isinstance(data, dict):
            break

        logger.warning(f"Decompose: non-dict response (attempt {attempt+1}), retrying…")
        data = None

    if not data:
        return _error_result(user_prompt)

    # Coerce operations into CSGOperation instances
    raw_ops = data.get("operations", [])
    operations: List[CSGOperation] = []
    for op in raw_ops:
        try:
            # Ensure position is always a list of 3 floats
            pos = op.get("position") or [0.0, 0.0, 0.0]
            if len(pos) < 3:
                pos = (pos + [0.0, 0.0, 0.0])[:3]

            operations.append(CSGOperation(
                op=op.get("op", "add"),
                shape=op.get("shape", "box"),
                params={
                    k: (int(v) if k in ("n_points", "sides") and isinstance(v, (int, float))
                        else float(v) if isinstance(v, (int, float))
                        else v)
                    for k, v in op.get("params", {}).items()
                },
                position=[float(v) for v in pos],
                rotation=op.get("rotation"),
                copies=op.get("copies"),
                copy_angle_step=op.get("copy_angle_step"),
            ))
        except Exception as e:
            logger.warning(f"Skipping malformed CSG operation: {e} — {op}")

    confidence = float(data.get("confidence", 0.0))

    return DecomposedObject(
        object_name=str(data.get("object_name", "Unknown Part")),
        real_world_reference=str(data.get("real_world_reference", "")),
        symmetry=data.get("symmetry"),
        estimated_dimensions=data.get("estimated_dimensions") or {},
        dims_are_estimated=bool(data.get("dims_are_estimated", True)),
        operations=operations,
        modifiers=list(data.get("modifiers") or []),
        confidence=max(0.0, min(1.0, confidence)),
        clarification_needed=bool(data.get("clarification_needed", False)),
        clarification_question=data.get("clarification_question"),
    )


def _error_result(user_prompt: str) -> DecomposedObject:
    """Return a safe fallback when decomposition fails."""
    return DecomposedObject(
        object_name="Unknown Part",
        real_world_reference=user_prompt[:80],
        confidence=0.0,
        clarification_needed=True,
        clarification_question="Could you describe the shape in more detail? (e.g. 'a flat disc with three arms')",
    )


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, json
    os.environ.setdefault("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

    for prompt in ["triangular fidget spinner", "makeup sponge"]:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        result = decompose_prompt(prompt)
        print(json.dumps(result.model_dump(), indent=2))
