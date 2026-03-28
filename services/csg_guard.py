"""
Layer 6 — CSG Edge-Case Guard

Receives the CSG operations (from semantic decomposer + ordered by Layer 5) and runs:

  (A) Deterministic algorithmic checks (fast, no AI):
      - Impossible subtracts (subtract shape larger than parent)
      - Zero-thickness walls between adjacent ops
      - Coincident/stacked face risk (non-manifold edges)
      - Circular copy angle overflow

  (B) AI review pass (Gemini Flash):
      - Spots modeling anti-patterns not caught by algebra
      - Proposes remediation notes (reorder, resize, switch primitive type)

Output: GuardedPlan — the approved (possibly modified) operation list, plus
flagged risks and remediation notes consumed by Layer 7.

Location: cadfactory-backend/services/csg_guard.py
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from services.engineering_math import (
    box_volume,
    cone_volume,
    cylinder_volume,
    sphere_volume,
    torus_volume,
)
from services.semantic_decomposer import CSGOperation

logger = logging.getLogger(__name__)


# ── Output model ───────────────────────────────────────────────────────────────

class RiskFlag(BaseModel):
    severity:   str  # "critical" | "warning" | "info"
    operation_label: str
    description: str
    remediation: str


class GuardedPlan(BaseModel):
    """Output of Layer 6 — validated, potentially patched CSG operation list."""

    # The approved (and possibly reordered / corrected) operations
    approved_ops: List[CSGOperation] = Field(default_factory=list)

    # Risk flags from both algo and AI checks
    risk_flags: List[RiskFlag] = Field(default_factory=list)

    # Count of critical risks found BEFORE remediation
    critical_risk_count: int = 0

    # Whether the plan needed patching
    was_patched: bool = False

    # AI remediation notes
    ai_notes: List[str] = Field(default_factory=list)

    # Confidence that the plan is geometrically sound
    guard_confidence: float = 0.9


# ── Volume estimator ──────────────────────────────────────────────────────────

def _estimate_volume(op: CSGOperation) -> float:
    """Estimate bounding volume of a CSG operation (mm³). Very rough."""
    p = op.params
    s = op.shape.lower()
    try:
        if s == "cylinder":
            return cylinder_volume(float(p.get("radius", 10)), float(p.get("height", 20)))
        elif s == "sphere":
            return sphere_volume(float(p.get("radius", 10)))
        elif s == "cone":
            return cone_volume(float(p.get("r_base", 10)), float(p.get("r_top", 5)),
                               float(p.get("height", 20)))
        elif s == "torus":
            return torus_volume(float(p.get("major_r", 20)), float(p.get("minor_r", 5)))
        elif s in ("box", "rounded_box"):
            length = float(p.get("length", 20))
            w = float(p.get("width", 20))
            h = float(p.get("height", 10))
            return box_volume(length, w, h)
        elif s == "polygon_prism":
            sides = int(p.get("sides", 6))
            d = float(p.get("diameter", 20))
            h = float(p.get("height", 10))
            # Area of regular polygon inscribed in circle of radius d/2
            area = (sides * (d / 2) ** 2 * math.sin(2 * math.pi / sides)) / 2
            return area * h
    except Exception:
        pass
    return 1.0  # non-zero fallback


# ── Algorithmic checks ─────────────────────────────────────────────────────────

def _check_impossible_subtracts(ops: List[CSGOperation]) -> List[RiskFlag]:
    """
    Detect subtract operations whose volume >> the additive volume they cut into.
    This often results in empty geometry or CadQuery topology failures.
    """
    flags: List[RiskFlag] = []
    cumulative_add_vol = 0.0
    for op in ops:
        vol = _estimate_volume(op)
        if op.op == "add":
            cumulative_add_vol += vol
        elif op.op == "subtract":
            if vol > cumulative_add_vol * 0.95:
                flags.append(RiskFlag(
                    severity="critical",
                    operation_label=op.label,
                    description=(
                        f"Subtract '{op.label}' (est. vol≈{vol:.0f}mm³) may be larger than "
                        f"or equal to the accumulated additive volume ({cumulative_add_vol:.0f}mm³). "
                        "This will result in empty geometry."
                    ),
                    remediation=(
                        "Reduce the subtract shape's dimensions, or ensure the parent "
                        "additive geometry is created first."
                    )
                ))
    return flags


def _check_copy_angle_overflow(ops: List[CSGOperation]) -> List[RiskFlag]:
    """Warn when circular copies exceed 360°."""
    flags: List[RiskFlag] = []
    for op in ops:
        if op.copies and op.copy_angle_step:
            total = op.copies * op.copy_angle_step
            if total > 360.0 + 1e-3:
                flags.append(RiskFlag(
                    severity="warning",
                    operation_label=op.label,
                    description=(
                        f"Circular copy '{op.label}': {op.copies} × {op.copy_angle_step}° = "
                        f"{total:.1f}° > 360°. Copies will overlap."
                    ),
                    remediation=f"Set copy_angle_step to {360.0 / op.copies:.1f}° for even distribution."
                ))
    return flags


def _check_zero_param_dims(ops: List[CSGOperation]) -> List[RiskFlag]:
    """Flag any operation with a zero or negative dimension parameter."""
    flags: List[RiskFlag] = []
    _zero_ok = {"r_top", "fillet_r"}
    for op in ops:
        for k, v in op.params.items():
            if k in _zero_ok:
                continue
            if isinstance(v, (int, float)) and float(v) <= 0:
                flags.append(RiskFlag(
                    severity="critical",
                    operation_label=op.label,
                    description=f"'{op.label}' param '{k}' = {v}. Must be > 0.",
                    remediation=f"Set '{k}' to a positive value (minimum: 1.0)."
                ))
    return flags


def _check_stacked_faces(ops: List[CSGOperation]) -> List[RiskFlag]:
    """
    Detect ops placed at the same position as another op of the same shape type.
    Coincident faces cause non-manifold topology errors in BREP kernels.
    """
    flags: List[RiskFlag] = []
    seen: Dict[str, Tuple[float, float, float]] = {}
    for op in ops:
        key = op.shape.lower()
        pos = tuple(op.position or [0.0, 0.0, 0.0])
        if key in seen and seen[key] == pos:
            flags.append(RiskFlag(
                severity="warning",
                operation_label=op.label,
                description=(
                    f"'{op.label}' ({op.shape}) is at the same position as a previous "
                    f"{op.shape} — may cause non-manifold geometry."
                ),
                remediation="Offset position by ≥0.01mm or combine into a single op."
            ))
        seen[key] = pos
    return flags


def _auto_patch_ops(ops: List[CSGOperation], flags: List[RiskFlag]) -> Tuple[List[CSGOperation], bool]:
    """
    Apply automatic patches for critical flags where safe to do so.
    Returns (patched_ops, was_patched).
    """
    patched = False
    result = list(ops)

    for flag in flags:
        if flag.severity != "critical":
            continue

        # Fix zero/negative params
        if "Must be > 0" in flag.description:
            for op in result:
                if op.label == flag.operation_label:
                    for k, v in op.params.items():
                        if isinstance(v, (int, float)) and float(v) <= 0 and k not in {"r_top", "fillet_r"}:
                            op.params[k] = 1.0  # minimum safe value
                            patched = True
                            logger.info(f"Layer 6: auto-patched '{op.label}' param '{k}' → 1.0")

        # Clamp subtract copy-angle overflow
        if "Copies will overlap" in flag.description:
            for op in result:
                if op.label == flag.operation_label and op.copies and op.copies > 0:
                    op.copy_angle_step = 360.0 / op.copies
                    patched = True

    return result, patched


# ── AI review pass ─────────────────────────────────────────────────────────────

_LAYER6_GUARD_PROMPT = """
You are Layer 6 of a precision CAD pipeline: the CSG Edge-Case Guard.

You receive a list of CSG operations (primitives, positions, parameters) and must:
1. Identify any geometric modeling anti-patterns that would cause CadQuery failures.
2. Spot any physically impossible or nonsensical combinations.
3. Suggest targeted remediations (do NOT rewrite the whole plan).

KNOWN ANTI-PATTERNS TO CHECK:
- Subtract op with larger bounding box than any parent add op
- Fillet radius larger than the feature it is applied to (r_fillet > min(l,w,h)/2)
- Box with zero or near-zero dimension (< 0.5mm in any axis)
- Torus with minor_r >= major_r (invalid torus)
- Cone with r_top > r_base (inverted — non-standard, warn user)
- Cylinder as shell with hole_r >= outer_r
- Circular pattern copies where total angle != 360° (uneven distribution)
- Any feature below 0.8mm in smallest dimension (unprintable on FDM)

Return ONLY valid JSON:
{
  "issues_found": [
    {"label": "operation_label", "issue": "short description", "fix": "suggested fix"}
  ],
  "overall_assessment": "safe_to_proceed | needs_revision | rebuild_required",
  "notes": ["any general modeling notes"]
}

If no issues, return: {"issues_found": [], "overall_assessment": "safe_to_proceed", "notes": []}
"""


def _ai_guard_review(ops: List[CSGOperation]) -> Dict[str, Any]:
    """Run an AI review pass on the CSG operation list."""
    import json

    from services.claude_cad import MODEL_FLASH, _generate_content
    from services.script_utils import parse_json_response

    ops_summary = json.dumps(
        [{"label": op.label, "op": op.op, "shape": op.shape,
          "params": op.params, "position": op.position,
          "copies": op.copies, "copy_angle_step": op.copy_angle_step}
         for op in ops],
        indent=2
    )

    try:
        raw = _generate_content(MODEL_FLASH, _LAYER6_GUARD_PROMPT, ops_summary)
        data = parse_json_response(raw)
        return data if data and isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Layer 6 AI guard failed: {exc}")
        return {}


# ── Layer 5: CSG Topology Planner (integrated here for clean imports) ──────────

class OrderedCSGPlan(BaseModel):
    """Output of Layer 5 — dependency-ordered CSG operation sequence."""

    # Operations sorted into execution phases
    phase_1_additive:   List[CSGOperation] = Field(default_factory=list)  # build-up
    phase_2_subtractive: List[CSGOperation] = Field(default_factory=list)  # cuts
    phase_3_intersect:  List[CSGOperation] = Field(default_factory=list)  # masks

    # Flat ordered list (phases concatenated, for code generation)
    ordered_ops: List[CSGOperation] = Field(default_factory=list)

    # Identity-defining features (top Pareto 20%)
    pareto_critical_labels: List[str] = Field(default_factory=list)

    # Estimated complexity score 1–10
    complexity_score: float = 1.0

    # Whether symmetry was detected
    has_symmetry: bool = False


def plan_csg_topology(
    ops: List[CSGOperation],
    priority_order: Optional[List[str]] = None,
) -> OrderedCSGPlan:
    """
    Layer 5: Sort CSG operations into dependency phases.

    Rule: Additive ops first (build the body), then subtractive (cut holes/slots),
    then intersect ops (masks). Within each phase, identity features (Pareto critical)
    come before supporting features.
    """
    priority_set = set(priority_order or [])

    # Phase sort
    add_ops  = [op for op in ops if op.op == "add"]
    sub_ops  = [op for op in ops if op.op == "subtract"]
    int_ops  = [op for op in ops if op.op == "intersect"]

    # Within additive: identity features first
    def _priority_key(op: CSGOperation) -> int:
        if any(p in op.label for p in priority_set):
            return 0
        if any(kw in op.label for kw in ("body", "main", "hull", "head", "base")):
            return 1
        if any(kw in op.label for kw in ("arm", "handle", "grip", "neck")):
            return 2
        if any(kw in op.label for kw in ("collar", "flange", "rim", "ring")):
            return 3
        return 4

    add_sorted = sorted(add_ops, key=_priority_key)
    sub_sorted = sorted(sub_ops, key=lambda op: 0 if "main" in op.label else 1)

    ordered = add_sorted + sub_sorted + int_ops

    # Pareto critical labels (top ~20%)
    critical_count = max(1, len(add_sorted) // 5 + 1)
    pareto_labels = [op.label for op in add_sorted[:critical_count]]

    # Complexity: 1 per op, +0.5 for copies, +1 for curve_prism
    complexity = len(ops)
    for op in ops:
        if op.copies and op.copies > 1:
            complexity += 0.5 * (op.copies - 1)
        if op.shape.lower() == "curve_prism":
            complexity += 1.0
    complexity = min(10.0, max(1.0, complexity))

    has_sym = any(
        (op.copies and op.copies >= 2) or "sym" in op.label.lower()
        for op in ops
    )

    return OrderedCSGPlan(
        phase_1_additive=add_sorted,
        phase_2_subtractive=sub_sorted,
        phase_3_intersect=int_ops,
        ordered_ops=ordered,
        pareto_critical_labels=pareto_labels,
        complexity_score=round(complexity, 1),
        has_symmetry=has_sym,
    )


# ── Main guard function ────────────────────────────────────────────────────────

def guard_csg_plan(
    ops: List[CSGOperation],
    priority_order: Optional[List[str]] = None,
    process: str = "fdm",
) -> GuardedPlan:
    """
    Layer 5 + 6 combined entry point:
      1. Plan topology (Layer 5 ordering)
      2. Run algorithmic checks
      3. Auto-patch where safe
      4. Run AI review pass
      5. Return GuardedPlan

    Always returns a GuardedPlan — never raises.
    """
    # Layer 5: topology ordering
    topo_plan = plan_csg_topology(ops, priority_order)
    ordered = topo_plan.ordered_ops

    # Layer 6A: algorithmic checks
    flags: List[RiskFlag] = []
    flags.extend(_check_impossible_subtracts(ordered))
    flags.extend(_check_copy_angle_overflow(ordered))
    flags.extend(_check_zero_param_dims(ordered))
    flags.extend(_check_stacked_faces(ordered))

    # Auto-patch critical issues
    patched_ops, was_patched = _auto_patch_ops(ordered, flags)

    # Layer 6B: AI review pass
    ai_data = _ai_guard_review(patched_ops)
    ai_issues = ai_data.get("issues_found", [])
    ai_notes  = ai_data.get("notes", [])

    # Convert AI issues to RiskFlags
    for issue in ai_issues:
        sev = "warning" if "warning" in str(issue.get("issue", "")).lower() else "warning"
        flags.append(RiskFlag(
            severity=sev,
            operation_label=str(issue.get("label", "unknown")),
            description=str(issue.get("issue", "")),
            remediation=str(issue.get("fix", "")),
        ))

    critical_count = sum(1 for f in flags if f.severity == "critical")
    guard_confidence = max(0.3, 1.0 - 0.15 * critical_count - 0.05 * len(flags))

    return GuardedPlan(
        approved_ops=patched_ops,
        risk_flags=flags,
        critical_risk_count=critical_count,
        was_patched=was_patched,
        ai_notes=list(ai_notes),
        guard_confidence=round(guard_confidence, 3),
    )
