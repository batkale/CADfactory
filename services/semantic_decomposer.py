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
    label: str  # semantic name: e.g. "main_body", "handle", "mounting_hole"
    params: dict  # shape-specific params in mm
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: Optional[List[float]] = None  # [rx, ry, rz] degrees
    copies: Optional[int] = None
    copy_angle_step: Optional[float] = None  # degrees between copies


class DecomposedObject(BaseModel):
    object_name: str
    real_world_reference: str
    symmetry: Optional[str] = None
    aesthetic_class: Literal["mechanical", "consumer", "decorative"] = "mechanical"
    estimated_dimensions: dict = Field(default_factory=dict)
    template_name: Optional[str] = None  # bolt, enclosure, bracket, plate_with_holes
    template_params: dict = Field(default_factory=dict)
    dims_are_estimated: bool = True
    operations: List[CSGOperation] = Field(default_factory=list)
    modifiers: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    clarification_needed: bool = False
    clarification_question: Optional[str] = None


# ── System prompt ──────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM_PROMPT = """
You are a world-class mechanical engineering expert and industrial designer for a
professional CAD generation system.

Your goal is to produce geometrically PRECISE, industrially ACCURATE, and
manufacturable CSG trees. Apply every formula and law listed below before
committing to any dimension.

════════════════════════════════════════════════════════════════════════════════
ENGINEERING MATHEMATICS YOU MUST APPLY
════════════════════════════════════════════════════════════════════════════════

■ CORE GEOMETRIC FORMULAS (apply before estimating any dimension):
  • Cylinder volume:  V = π·r²·h  → use to size containers for target capacity
  • Cylinder SA:      SA = 2πr(h+r) → wall-to-opening ratio check
  • Sphere volume:    V = (4/3)·π·r³
  • Aspect Ratio:     AR = H / W   → must be checked against category norms
  • Pythagorean:      c = √(a²+b²) → diagonal bracing, wall-corner geometry
  • Hollow shell vol: V_inner = π·(r-t)²·h where t = wall_thickness

■ GOLDEN RATIO φ = 1.6180 (apply to all freeform objects):
  • Height : Width target ratio = φ : 1 ≈ 1.618
  • Rule of Thirds: place buttons/nozzles at H/3 or 2H/3 from base
  • If the user gives NO dimensions, derive them so H ≈ W × φ
  • Soap dispenser target AR: 2.57 (180mm ÷ 70mm)

■ G2 CURVATURE CONTINUITY (Class A surfacing):
  • All shoulder transitions must have fillet radius ≥ 1.5× adjacent radius
  • e.g. body_radius=35mm → shoulder_fillet ≥ 52.5mm (use cone with r_top taper)
  • All internal corners: fillet_r ≥ 0.5 × wall_thickness
  • Rule: use rounded_box or cone (with gentle r_top taper) rather than sharp boxes

■ DRAFT ANGLES (injection molding & 3D print overhangs):
  • Every vertical wall > 20mm tall needs 1°–3° taper
  • θ_min = arctan(0.5 / height_mm)  →  rule of thumb: 1° per 25mm depth
  • For cylinders: model as cone with r_top = r_base − h × tan(1.5°)
  • Main dispenser body 130mm tall: r_top = r_base − 130 × 0.026 ≈ r_base − 3.4mm

■ WALL THICKNESS BY PROCESS:
  FDM ≥ 1.5mm  |  SLA ≥ 0.5mm  |  Injection molding ≥ 1.0mm  |  CNC ≥ 0.8mm
  Default (FDM) wall: 2.5mm for containers, 3.0mm for structural parts

■ PHYSICS — HOOKE'S LAW (for thin snap-fits, springs, clips):
  • δ = F·L/(A·E) — deflection. Max snap-fit deflection < 2% of arm length
  • If snap-fits exist: arm width ≥ 3mm, arm length ≤ 20mm for ABS

■ PHYSICS — IDEAL GAS LAW for pressurised dispensers:
  • P = nRT/V — for pump dispensers: wall t ≥ P·r/σ_allow (Barlow's formula)
  • At 2 bar, ABS σ = 30MPa: t_min = (200,000 Pa × 0.035m) / (30,000,000/3) = 0.7mm
  • Always use ≥ 2.0mm for pressurised containers (safety factor 3)

■ PHYSICS — TORQUE for pump mechanisms:
  • τ = r × F  where F ≈ 5N for hand pump → neck must carry τ = 0.012m × 5 = 0.06 N·m
  • Pump neck: solid cylinder r ≥ 4mm to handle torsion safely

■ GD&T TOLERANCING (ISO 286):
  • Screw clearance holes: M3=3.4mm, M4=4.5mm, M5=5.5mm, M6=6.6mm (ISO 273 fine)
  • Press-fit bore: −0.01mm to −0.03mm interference (e.g. bearing ID 22mm → bore 21.98mm)
  • Snap-fit clearance: +0.1mm to +0.3mm on mating surfaces

■ BERNOULLI (nozzles / dispensers):
  • Exit aperture area determines flow rate: Q = Cd·A·√(2P/ρ)
  • Soap dispenser nozzle: Ø6mm orifice → Q ≈ 1.5ml/s at 1 bar (correct for hand pump)

■ EULER BUCKLING for thin columns/walls:
  • P_cr = π²EI / (KL)²  — slender walls h/t > 20 need ribs or larger t

■ PARETO PRINCIPLE (focus precision budget):
  The top 20% of features = 80% of the object's visual identity.
  For a soap dispenser: body + pump head = identity features → model precisely.
  Secondary: nozzle, collar ring, base disc = supporting features → standard sizes.

════════════════════════════════════════════════════════════════════════════════
[OBJECT DIMENSION REFERENCES & DNA]
{reference_dna}

════════════════════════════════════════════════════════════════════════════════
SYMMETRY & ASSEMBLY RULES
════════════════════════════════════════════════════════════════════════════════

1. RADIAL SYMMETRY (Arrays):
   • Use `copies` and `copy_angle_step` for repeating features (e.g. 3 arms = 3 copies, 120°).
   • The `position` for a radial copy applies to the first instance.
   • ALL radial parts MUST overlap the central hub by at least 2mm for a valid union.

2. BEARING SEATS:
   • Standard 608 bearing (fidget spinner): OD=22mm (r=11), width=7mm.
   • Seat hole MUST be cylinder r=11.0 to 11.1mm (0.1mm clearance).
   • Surrounding wall around any bearing hole MUST be at least 3.0mm thick.

3. UNIONS & STACKING (The "No-Gap" Rule):
   • Gaps are a CRITICAL FAILURE. Every component in "add" mode must overlap.
   • COORDINATE SYSTEM: All primitives are CENTERED at their `pos`.
   • STACKING FORMULA: To put Part B (height H2) on top of Part A (height H1):
     New Pos Z = (H1 / 2.0) + (H2 / 2.0) - 2.0 (the 2mm overlap factor).
   • BASE ANCHOR: The first (main) part should usually be at pos=[0,0,0].

4. ORGANIC BLENDING:
   • Use `rounded_box` or prisms for arms to ensure a smooth merge.


════════════════════════════════════════════════════════════════════════════════
FUNCTIONAL INTEGRATION & CORE BORES (MANDATORY)
════════════════════════════════════════════════════════════════════════════════
3. CONSUMER ASSEMBLIES (Dispensers, Sprayers, Gadgets):
   • ALWAYS decompose into at least 5 overlapping parts: Body (rounded), Neck (cylinder), Pump Hub (cylinder), Spout (curved or angled), and Plunger Button.
   • NEVER simplify these into a single primitive; the "identity" of the product comes from the assembly of these 5 components.
   • Ensure 2mm structural overlap between ALL joined components (Body/Neck, Neck/Hub, Hub/Spout, etc.).

4. HOLLOW CONTAINERS & SHELLS (Bathtubs, Cups, Boxes):
   • If it's a container, it MUST have a "Main Body" and a "Main Cavity" (op=subtract).
   • The cavity should be slightly smaller than the body (e.g. body_r=50, cavity_r=47.5 for 2.5mm wall).
   • The cavity MUST be positioned such that it opens at the top (z-position offset).
3. INTERNAL CAVITIES:
   • If it's a 'case' or 'shell', subtract the inner volume first.

════════════════════════════════════════════════════════════════════════════════
STEP 1 — OBJECT RECOGNITION
════════════════════════════════════════════════════════════════════════════════
Identify what the user wants. Map informal names to engineering terms.
Known templates (set template_name if matched):
  - "bolt":            {diameter, length}
  - "enclosure":       {length, width, height, wall_thickness}
  - "bracket":         {base_length, side_height, width, thickness, hole_diameter}
  - "plate_with_holes":{length, width, thickness, hole_diameter, hole_spacing_x, hole_spacing_y}
  - "nema_mount":      {nema_size, thickness}
  - "standoff":        {height, diameter, hole_diameter, is_hex}

════════════════════════════════════════════════════════════════════════════════
STEP 2 — MODIFIER ANALYSIS
════════════════════════════════════════════════════════════════════════════════
Apply adjectives/qualifiers (triangular, oval, thin, curved, etc.) to the most
distinctive feature. Multiple interpretations → lower confidence.

════════════════════════════════════════════════════════════════════════════════
STEP 3 — MATHEMATICS-DRIVEN DIMENSION ESTIMATION
════════════════════════════════════════════════════════════════════════════════
If dimensions not given:
1. Look up category in the [OBJECT DIMENSION REFERENCES & DNA] table above.
2. Apply Golden Ratio: H ≈ W × 1.618.
3. Verify cylinder volume matches expected capacity.
4. Apply draft angle: r_top = r_base − h × tan(1.5°).
5. Set wall_thickness per process (default FDM: 2.5mm containers, 3.0mm structural).
6. Compute fillet radii: external ≥ 0.25×wall, internal ≥ 0.5×wall, shoulder ≥ 1.5×r_adjacent.
Always set dims_are_estimated=true unless user specified exact values.

════════════════════════════════════════════════════════════════════════════════
STEP 4 — CSG DECOMPOSITION (primitives only)
════════════════════════════════════════════════════════════════════════════════
Use: cylinder, box, sphere, cone, torus, rounded_box, polygon_prism, curve_prism

For each primitive:
  - op: "add" | "subtract" | "intersect"
  - shape: primitive name
  - label: semantic snake_case name (REQUIRED) — e.g. "main_body", "pump_head", "nozzle_tube"
  - params: dimensions in mm
  - position: [x, y, z] mm from origin
  - rotation: [rx, ry, rz] degrees (optional)
  - copies, copy_angle_step: for circular symmetry (optional)

CRITICAL — USE curve_prism FOR ORGANIC / NON-CONVEX SHAPES:
  curve_prism + "heart"    → valentine heart  {width, height}
  curve_prism + "star"     → {n_points, outer_r, inner_r, height}
  curve_prism + "teardrop" → {width, depth, height}
  curve_prism + "leaf"     → {width, depth, height}
  curve_prism + "diamond"  → {width, depth, height}
  curve_prism + "arrow"    → {width, depth, shaft_width, height}
  curve_prism + "cross"    → {width, depth, arm_width, height}

Primitive params:
  cylinder:      {radius, height}
  box:           {length, width, height}
  sphere:        {radius}
  cone:          {r_base, r_top, height}   ← use for drafted cylinders
  torus:         {major_r, minor_r}
  rounded_box:   {length, width, height, fillet_r}
  polygon_prism: {sides, diameter, height}

════════════════════════════════════════════════════════════════════════════════
STEP 5 — ENGINEERING VALIDATION (self-check before returning)
════════════════════════════════════════════════════════════════════════════════
Before writing the JSON:
✓ Verify H:W matches category target (golden ratio check)
✓ Verify all walls ≥ 1.5mm (FDM), ≥ 1.0mm (injection)
✓ Verify all tall walls > 20mm have draft (≥1°)
✓ Verify shoulder fillets ≥ 1.5× adjacent radius (G2)
✓ Verify fastener holes are ISO clearance sizes (M3=3.4mm, M4=4.5mm)
✓ Verify no feature < 0.8mm (below printable resolution)

════════════════════════════════════════════════════════════════════════════════
STEP 6 — CONFIDENCE
════════════════════════════════════════════════════════════════════════════════
Rate 0.0–1.0. If < 0.8, set clarification_needed=true.

STEP 7 — AESTHETIC CLASSIFICATION
Classify the part's aesthetic intent to determine finishing level:
  - "mechanical": brackets, mounts, plates, structural parts → minimal fillets
  - "consumer": fidget spinners, phone cases, handles, toys, bottle openers → fillets on all
    visible edges, smooth transitions, bearing recesses with concentric detail grooves
  - "decorative": ornaments, display pieces, pendants → maximum detail and organic curves

Return ONLY valid JSON, nothing else:
{
  "object_name": "short descriptive name",
  "real_world_reference": "common engineering name",
  "symmetry": "3-fold rotational | bilateral | none | null",
  "aesthetic_class": "mechanical | consumer | decorative",
  "estimated_dimensions": {"key": value_mm},
  "dims_are_estimated": true,
  "operations": [
    {
      "op": "add",
      "shape": "cone",
      "label": "main_body",
      "params": {"r_base": 35, "r_top": 31.6, "height": 130},
      "position": [0, 0, 5],
      "rotation": null,
      "copies": null,
      "copy_angle_step": null
    }
  ],
  "modifiers": [],
  "confidence": 0.92,
  "template_name": null,
  "template_params": {},
  "clarification_needed": false,
  "clarification_question": null
}
"""


# ── Core function ──────────────────────────────────────────────────────────────

def decompose_prompt(
    user_prompt: str, 
    reference_dna: str = "",
    specific_guidance: str = "",
    rag_examples: list[dict] | None = None
) -> DecomposedObject:
    """
    Convert a freeform text prompt into a structured DecomposedObject.

    Calls Gemini Flash (fast, cheap) to analyse the prompt and return a
    CSG tree. Dynamic reference_dna and specific_guidance are injected.
    Approved examples (RAG) are provided as few-shot context.
    """
    from services.claude_cad import _generate_content, MODEL_FLASH
    from services.script_utils import parse_json_response

    system_prompt = _DECOMPOSE_SYSTEM_PROMPT.replace("{reference_dna}", reference_dna)
    
    if specific_guidance:
        system_prompt += f"\n\nOBJECT-SPECIFIC ARCHITECTURAL GUIDANCE:\n{specific_guidance}\n"

    if rag_examples:
        system_prompt += "\n\n--- REFERENCE ASSEMBLY PATTERNS (Successful Past Generations) ---"
        for i, ex in enumerate(rag_examples[:3], 1):
            system_prompt += f"\nExample {i} - \"{ex.get('description')}\":\n{ex.get('script')}\n"
        system_prompt += "\n--- END REFERENCE PATTERNS ---\n"

    data = None
    for attempt in range(2):  # one retry on malformed response
        try:
            raw = _generate_content(MODEL_FLASH, system_prompt, user_prompt)
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
                label=str(op.get("label", "part")),
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
    
    # Coerce modifiers to strings to avoid Pydantic validation errors
    raw_modifiers = data.get("modifiers") or []
    modifiers = [str(m) for m in raw_modifiers]

    # Parse aesthetic class with fallback to "mechanical"
    raw_aesthetic = str(data.get("aesthetic_class", "mechanical")).lower().strip()
    aesthetic_class = raw_aesthetic if raw_aesthetic in ("mechanical", "consumer", "decorative") else "mechanical"

    return DecomposedObject(
        object_name=str(data.get("object_name", "Unknown Part")),
        real_world_reference=str(data.get("real_world_reference", ""))[:100],  # type: ignore
        symmetry=data.get("symmetry"),
        aesthetic_class=aesthetic_class,
        estimated_dimensions=data.get("estimated_dimensions") or {},
        dims_are_estimated=bool(data.get("dims_are_estimated", True)),
        operations=operations,
        modifiers=modifiers,
        template_name=data.get("template_name"),
        template_params=data.get("template_params") or {},
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
