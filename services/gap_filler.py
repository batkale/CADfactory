"""
Layer 4 — Generative Gap-Filling & Spec Completion

Takes the math-validated parameter set from Layer 3 and uses Gemini Flash to
infer ALL missing but necessary mechanical parameters:

  - Wall thickness (if not explicitly set)
  - Draft angles for tall walls
  - Fillet radii (internal vs external vs G2-class)
  - GD&T tolerance class
  - Material guess (if not stated)
  - Surface finish classification
  - Snap-fit / press-fit clearances
  - Ribs/gussets recommendation for tall thin walls

Output is a `RefinedSpec` — the complete, fully-populated specification that
feeds directly into Layer 5 (CSG Topology Planning).

Location: cadfactory-backend/services/gap_filler.py
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from services.nlp_extractor import ContextBundle
from services.param_sorter import SortedParams
from services.constraint_validator import ValidationResult
from services.semantic_decomposer import CSGOperation

logger = logging.getLogger(__name__)


# ── Output model ───────────────────────────────────────────────────────────────

class RefinedSpec(BaseModel):
    """
    Fully-populated design specification after all 4 input layers.
    This is the hand-off object to Layer 5 (CSG Topology Planner).
    """

    # Identity
    object_name:          str = "Unknown Part"
    object_category:      str = "other"
    real_world_reference: str = ""
    manufacturing_process: str = "fdm"

    # Dimensions (mm) — merged from Layers 1–4
    width_mm:   float = 0.0
    depth_mm:   float = 0.0
    height_mm:  float = 0.0
    radius_mm:  float = 0.0  # primary radius (if cylindrical)

    # Engineering parameters — filled by gap filler if not in explicit/implied
    wall_mm:             float = 2.5
    draft_angle_deg:     float = 1.5
    fillet_r_external:   float = 2.0
    fillet_r_internal:   float = 1.25
    fillet_r_shoulder:   float = 4.0
    tolerance_class:     str = "IT7"  # ISO 286 grade
    material_guess:      str = "ABS"
    surface_finish:      str = "smooth"
    fit_type:            Optional[str] = None  # "press" | "sliding" | "snap" | null

    # Hole / fastener info
    hole_diameter_mm:    Optional[float] = None
    hole_count:          int = 0
    thread_nominal_mm:   Optional[float] = None
    thread_pitch_mm:     Optional[float] = None

    # Volume / capacity
    capacity_ml:         Optional[float] = None

    # Symmetry
    symmetry:            Optional[str] = None  # "bilateral" | "3-fold rotational" | "none"

    # Geometric modifiers from Layer 1
    geometric_modifiers: List[str] = Field(default_factory=list)

    # Situational notes from Layer 1 (forwarded for Layer 5/6 context)
    situational_notes:   List[str] = Field(default_factory=list)

    # Priority feature order from Layer 2
    priority_order:      List[str] = Field(default_factory=list)

    # Layer 3 corrected params (full dict for pass-through)
    validated_params:    Dict[str, Any] = Field(default_factory=dict)

    # CSG operations from decomposer (forwarded)
    operations:          List[CSGOperation] = Field(default_factory=list)

    # Aggregate confidence (product of all layer confidences)
    confidence:          float = 0.0

    # Gap-fill notes — what was inferred vs. what was explicit
    inferred_fields:     List[str] = Field(default_factory=list)

    # Whether clarification is needed
    clarification_needed: bool = False
    clarification_question: Optional[str] = None


# ── Layer 4 system prompt ──────────────────────────────────────────────────────

_LAYER4_GAP_FILL_PROMPT = """
You are Layer 4 of a precision CAD pipeline: the Generative Gap-Filler.

You receive a partially-specified design parameter set and must infer ALL missing
but mechanically necessary values. Apply real-world mechanical engineering rules.

════════════════════════════════════════════════════════════════════════════════
WHAT TO INFER (fill any field that is null or missing):
════════════════════════════════════════════════════════════════════════════════

1. wall_mm
   Default: 2.5mm for FDM containers, 3.0mm for structural, 1.5mm for SLA.
   Pressure vessels: use Barlow's formula. Always ≥ process minimum.

2. draft_angle_deg
   Walls > 20mm tall: minimum 1.5°. Injection molding: 2–3°. FDM: 1°.
   Formula: θ = 1° per 25mm depth. Minimum: 1.0°.

3. fillet_r_external
   External edges: ≥ 0.25 × wall_mm. Aesthetic minimum: 1.5mm.

4. fillet_r_internal
   Internal corners: ≥ 0.5 × wall_mm to prevent stress concentration.

5. fillet_r_shoulder  (G2 continuity)
   Shoulder transitions: ≥ 1.5 × adjacent radius for Class A surface smoothness.

6. tolerance_class
   General engineering: IT7. High precision: IT5. Rough: IT10.
   Snap-fits: IT8. Press fits: IT6. Thread clearances: IT9.

7. material_guess
   Consumer products: ABS or PETG. Medical: Nylon or Polycarbonate.
   Structural: ABS or Al6061 (if CNC). Flexible: TPU. Default: ABS.

8. surface_finish
   "smooth" for consumer products. "textured" for grip areas. "raw" for hidden.

9. fit_type  (if fasteners or assembly joints present)
   Press fit: interference − 0.01 to −0.03mm. Slip fit: +0.2mm. Snap: +0.15mm.

10. symmetry
    Cylindrical thin body → "bilateral". With 3+ radial arms → "3-fold rotational".
    Irregular → "none".

════════════════════════════════════════════════════════════════════════════════
RULES:
- Only fill fields that are null or 0. Never override explicitly stated values.
- All dimensions in mm. Never invent impossible values.
- Return ONLY valid JSON — no markdown, no explanation.

JSON format:
{
  "wall_mm": 2.5,
  "draft_angle_deg": 1.5,
  "fillet_r_external": 2.0,
  "fillet_r_internal": 1.25,
  "fillet_r_shoulder": 4.0,
  "tolerance_class": "IT7",
  "material_guess": "ABS",
  "surface_finish": "smooth",
  "fit_type": null,
  "symmetry": "bilateral",
  "inferred_fields": ["wall_mm", "draft_angle_deg", "fillet_r_external"]
}
"""


# ── Core function ──────────────────────────────────────────────────────────────

def fill_gaps(
    context:    ContextBundle,
    sorted_p:   SortedParams,
    validated:  ValidationResult,
    decomposed_ops: Optional[List[CSGOperation]] = None,
) -> RefinedSpec:
    """
    Layer 4: Complete the design specification using AI gap-filling.

    Merges all upstream layer outputs into a unified RefinedSpec.
    Uses Flash for the gap-fill inference. Falls back to engineering defaults.
    """
    from services.claude_cad import _generate_content, MODEL_FLASH
    from services.script_utils import parse_json_response

    corrected = validated.auto_corrected_params or {}

    # ── Build the context message for AI ───────────────────────────────────────
    context_msg = (
        f"Object: {context.object_category} | Intent: {context.intent_type}\n"
        f"Manufacturing: {sorted_p.manufacturing_process}\n"
        f"Situational notes: {'; '.join(context.situational_notes[:3])}\n"
        f"Geometric modifiers: {', '.join(context.geometric_modifiers)}\n\n"
        f"Current validated parameters:\n"
    )
    for k, v in corrected.items():
        context_msg += f"  {k}: {v}\n"

    context_msg += "\nInfer ALL missing engineering parameters listed above."

    # ── AI call ────────────────────────────────────────────────────────────────
    inferred: Dict[str, Any] = {}
    inferred_fields: List[str] = []
    try:
        raw = _generate_content(MODEL_FLASH, _LAYER4_GAP_FILL_PROMPT, context_msg)
        data = parse_json_response(raw)
        if data and isinstance(data, dict):
            inferred = data
            inferred_fields = list(data.get("inferred_fields", []))
    except Exception as exc:
        logger.warning(f"Layer 4 gap-fill AI call failed: {exc} — using engineering defaults")
        inferred = _engineering_defaults(sorted_p.manufacturing_process)
        inferred_fields = list(inferred.keys())

    # ── Resolve dimensions (priority: corrected > explicit > implied > inferred) ─
    def _get(*keys: str, default=0.0):
        """Try keys in order of precedence."""
        for k in keys:
            v = corrected.get(k) or sorted_p.explicit_dims.get(k) or sorted_p.implied_dims.get(k)
            if v:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return default

    width_mm  = _get("width", "outer_diameter")
    depth_mm  = _get("depth")
    height_mm = _get("height", "inner_height")
    radius_mm = _get("radius", "inner_radius")

    # If only radius given, derive width
    if radius_mm > 0 and width_mm == 0.0:
        width_mm = radius_mm * 2.0
    # If only width given, derive radius
    if width_mm > 0 and radius_mm == 0.0:
        radius_mm = width_mm / 2.0

    # ── Map AI inferred values (don't override corrected) ─────────────────────
    def _merge(key: str, default: float) -> float:
        v = corrected.get(key) or sorted_p.explicit_dims.get(key) or inferred.get(key)
        if v:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        return default

    wall_mm           = _merge("wall_thickness", _merge("wall_mm", inferred.get("wall_mm", 2.5)))
    draft_deg         = float(inferred.get("draft_angle_deg", 1.5))
    fillet_ext        = float(inferred.get("fillet_r_external", 2.0))
    fillet_int        = float(inferred.get("fillet_r_internal", 1.25))
    fillet_shoulder   = float(inferred.get("fillet_r_shoulder", 4.0))
    tolerance_class   = str(inferred.get("tolerance_class", "IT7"))
    material          = context.material_hint or str(inferred.get("material_guess", "ABS"))
    surface_finish    = str(inferred.get("surface_finish", "smooth"))
    fit_type          = sorted_p.functional_params.get("fit_type") or inferred.get("fit_type")
    symmetry          = str(inferred.get("symmetry", "none")) if inferred.get("symmetry") else None

    hole_d  = _get("hole_diameter", "bore_diameter", "nozzle_diameter") or None
    hole_d  = hole_d if hole_d and hole_d > 0 else None
    hole_n  = int(sorted_p.functional_params.get("hole_count", 0))
    t_nom   = _get("thread_nominal_mm") or None
    t_nom   = t_nom if t_nom and t_nom > 0 else None
    t_pitch = _get("thread_pitch_mm") or None
    t_pitch = t_pitch if t_pitch and t_pitch > 0 else None
    cap_ml  = _get("capacity_ml") or None

    # ── Build object name from category + modifiers ───────────────────────────
    modifiers_str = " ".join(context.geometric_modifiers[:2])
    obj_name = f"{modifiers_str} {context.object_category}".strip().title()

    # ── Combine confidences from all 4 layers ─────────────────────────────────
    layer_conf = (
        context.confidence
        * sorted_p.confidence
        * validated.engineering_score
        * (0.85 if inferred else 0.60)   # gap-fill adds confidence
    )
    combined_confidence = round(max(0.05, min(1.0, layer_conf ** 0.4)), 3)
    # NOTE: using 4th-root to avoid confidence collapsing to near-zero

    return RefinedSpec(
        object_name=obj_name,
        object_category=context.object_category,
        real_world_reference=context.object_category,
        manufacturing_process=sorted_p.manufacturing_process,
        width_mm=round(width_mm, 3),
        depth_mm=round(depth_mm, 3),
        height_mm=round(height_mm, 3),
        radius_mm=round(radius_mm, 3),
        wall_mm=round(wall_mm, 3),
        draft_angle_deg=round(draft_deg, 2),
        fillet_r_external=round(fillet_ext, 3),
        fillet_r_internal=round(fillet_int, 3),
        fillet_r_shoulder=round(fillet_shoulder, 3),
        tolerance_class=tolerance_class,
        material_guess=material,
        surface_finish=surface_finish,
        fit_type=fit_type,
        hole_diameter_mm=hole_d,
        hole_count=hole_n,
        thread_nominal_mm=t_nom,
        thread_pitch_mm=t_pitch,
        capacity_ml=cap_ml,
        symmetry=symmetry,
        geometric_modifiers=context.geometric_modifiers,
        situational_notes=context.situational_notes,
        priority_order=sorted_p.priority_order,
        validated_params=corrected,
        operations=decomposed_ops or [],
        confidence=combined_confidence,
        inferred_fields=inferred_fields,
        clarification_needed=context.confidence < 0.5,
        clarification_question=context.clarification_question if context.confidence < 0.5 else None,
    )


def _engineering_defaults(process: str) -> Dict[str, Any]:
    """Fallback engineering defaults when AI gap-fill fails."""
    from services.engineering_math import MIN_WALL_THICKNESS
    min_wall = MIN_WALL_THICKNESS.get(process, 1.5)
    return {
        "wall_mm": max(2.5, min_wall),
        "draft_angle_deg": 1.5,
        "fillet_r_external": 2.0,
        "fillet_r_internal": 1.25,
        "fillet_r_shoulder": 4.0,
        "tolerance_class": "IT7",
        "material_guess": "ABS",
        "surface_finish": "smooth",
        "fit_type": None,
        "symmetry": None,
        "inferred_fields": ["wall_mm", "draft_angle_deg", "fillet_r_external",
                            "fillet_r_internal", "fillet_r_shoulder",
                            "tolerance_class", "material_guess"],
    }
