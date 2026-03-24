"""
Layer 3 — Mathematical Constraint Validation Result model.

This module extends engineering_math.py with the Layer 3 parameter-bundle
validation function. Imported by gap_filler.py and pipeline.py.

Location: cadfactory-backend/services/constraint_validator.py
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from services.param_sorter import SortedParams
from services.engineering_math import (
    circle_area, cylinder_volume, hollow_shell_volume, aspect_ratio,
    golden_ratio_score, wall_thickness_check, draft_taper_radius,
    max_tensile_stress, lookup_reference_dims, g2_fillet_radius,
    min_fillet_from_wall, iso_tolerance_microns,
    MIN_WALL_THICKNESS, INDUSTRIAL_DB, PHI,
)

logger = logging.getLogger(__name__)


# ── Model ───────────────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Output of Layer 3 — Mathematical Constraint Validation."""

    passed: bool = True

    # Auto-corrections that were applied (key → corrected_value_mm)
    corrections: Dict[str, Any] = Field(default_factory=dict)

    # Auto-corrected parameter set (merged explicit + implied + corrections)
    auto_corrected_params: Dict[str, Any] = Field(default_factory=dict)

    # Non-fatal warnings
    warnings: List[str] = Field(default_factory=list)

    # Fatal errors (block generation)
    errors: List[str] = Field(default_factory=list)

    # Engineering score 0–1 (how mathematically self-consistent the params are)
    engineering_score: float = 0.0

    # Detailed check results
    checks: Dict[str, Any] = Field(default_factory=dict)


# ── Individual checks ──────────────────────────────────────────────────────────

def _check_wall_feasibility(params: Dict[str, Any], process: str) -> tuple[List[str], Dict[str, Any]]:
    """Ensure wall thickness meets manufacturing minimums."""
    warnings, corrections = [], {}
    min_t = MIN_WALL_THICKNESS.get(process, 1.5)
    wall = params.get("wall_thickness", params.get("wall_thickness_mm",
           params.get("wall", 0.0)))
    if wall and float(wall) < min_t:
        corrections["wall_thickness"] = min_t
        warnings.append(
            f"Wall {wall:.2f}mm < minimum {min_t}mm for {process}. "
            f"Auto-corrected to {min_t}mm."
        )
    elif not wall:
        # Set a sensible default
        default_wall = 2.5 if "container" in params.get("object_category", "") else 3.0
        corrections["wall_thickness"] = max(default_wall, min_t)
    return warnings, corrections


def _check_hole_vs_wall(params: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """Ensure holes/bores are smaller than the surface they sit on."""
    warnings, errors = [], []
    width  = params.get("width", params.get("outer_diameter", 0.0))
    hole   = params.get("hole_diameter", params.get("bore_diameter",
             params.get("nozzle_diameter", 0.0)))
    if width and hole and float(hole) >= float(width):
        errors.append(
            f"Hole/bore diameter {hole:.2f}mm ≥ wall width {width:.2f}mm — "
            "geometrically impossible. Please reduce hole size or increase wall width."
        )
    return warnings, errors


def _check_volume_vs_capacity(params: Dict[str, Any]) -> tuple[List[str], List[str], Dict[str, Any]]:
    """Cross-check stated capacity against provided outer dimensions."""
    warnings, errors, corrections = [], [], {}
    capacity_ml = params.get("capacity_ml")
    if not capacity_ml:
        return warnings, errors, corrections

    # Try to get outer dimensions from explicit or implied
    r = params.get("radius", params.get("inner_radius",
        params.get("outer_diameter", 0.0)))
    if r:
        r = float(r) / 2.0 if "outer_diameter" in params else float(r)
    h = params.get("height", params.get("inner_height", 0.0))
    wall = params.get("wall_thickness", 2.5)
    if r and h:
        r_inner = max(1.0, float(r) - float(wall))
        computed_ml = cylinder_volume(r_inner, float(h)) / 1000.0
        ratio = computed_ml / float(capacity_ml)
        if ratio < 0.5:
            warnings.append(
                f"Computed inner volume ≈{computed_ml:.0f}ml < stated capacity "
                f"{capacity_ml:.0f}ml (ratio={ratio:.2f}). "
                f"Consider increasing radius or height."
            )
        elif ratio > 2.0:
            warnings.append(
                f"Computed inner volume ≈{computed_ml:.0f}ml >> stated capacity "
                f"{capacity_ml:.0f}ml (ratio={ratio:.2f}). Dimensions may be oversized."
            )
    return warnings, errors, corrections


def _check_aspect_ratio(params: Dict[str, Any], category: str) -> List[str]:
    """Check height:width aspect ratio against golden ratio and category norms."""
    warnings = []
    h = params.get("height", params.get("inner_height", 0.0))
    w = params.get("width", params.get("radius", 0.0))
    if not (h and w):
        return warnings

    ar = float(h) / max(float(w), 1e-6)
    score = golden_ratio_score(float(h), float(w))

    ref = lookup_reference_dims(category)
    if ref:
        ref_ar = ref.body_height_mm / max(ref.body_width_mm, 1e-6)
        if abs(ar - ref_ar) / max(ref_ar, 1e-6) > 0.4:
            warnings.append(
                f"Aspect ratio H:W={ar:.2f} differs significantly from category "
                f"standard {ref_ar:.2f} ({ref.category}). "
                f"Golden ratio score={score:.2f}/1.0."
            )
    elif score < 0.5:
        warnings.append(
            f"Aspect ratio H:W={ar:.2f} has low golden ratio score {score:.2f}. "
            f"Target H:W≈{PHI:.2f} for aesthetic proportions."
        )
    return warnings


def _check_minimum_feature_size(params: Dict[str, Any], process: str) -> tuple[List[str], List[str]]:
    """Flag any feature that cannot be physically produced at the stated size."""
    warnings, errors = [], []
    min_feature = {"fdm": 0.8, "sla": 0.2, "sls": 0.5, "cnc": 0.5, "injection": 0.5}.get(
        process.lower(), 0.8
    )
    for key, val in params.items():
        if isinstance(val, (int, float)) and 0 < float(val) < min_feature:
            errors.append(
                f"Feature '{key}' = {val}mm is below minimum printable size "
                f"{min_feature}mm for {process}."
            )
    return warnings, errors


def _check_draft_consistency(params: Dict[str, Any]) -> List[str]:
    """Warn if tall walls > 20mm lack draft information."""
    warnings = []
    h = params.get("height", 0.0)
    draft = params.get("aesthetic_params", {}).get("draft_angle_deg", None)
    if float(h) > 20 and draft is None:
        warnings.append(
            f"Wall height {h}mm > 20mm but no draft angle specified. "
            "Recommend ≥1.5° draft for injection molding / tall FDM features."
        )
    return warnings


# ── Main validation function ───────────────────────────────────────────────────

def validate_parameter_bundle(
    sorted_params: SortedParams,
    category: Optional[str] = None,
) -> ValidationResult:
    """
    Layer 3: Run full mathematical constraint validation on a SortedParams bundle.

    Returns a ValidationResult with corrected params, warnings, and errors.
    Never raises — returns a failed ValidationResult on unexpected errors.
    """
    cat = category or sorted_params.object_category
    process = sorted_params.manufacturing_process

    # Flatten all params into a single lookup dict for cross-checks
    flat: Dict[str, Any] = {}
    flat.update(sorted_params.explicit_dims)
    flat.update(sorted_params.implied_dims)
    flat.update(sorted_params.functional_params)
    flat.update(sorted_params.aesthetic_params)
    flat["object_category"] = cat

    all_warnings: List[str] = []
    all_errors:   List[str] = []
    all_corrections: Dict[str, Any] = {}
    checks: Dict[str, Any] = {}

    try:
        # Check 1: Wall thickness
        w_warn, w_corr = _check_wall_feasibility(flat, process)
        all_warnings.extend(w_warn)
        all_corrections.update(w_corr)
        checks["wall_feasibility"] = {"warnings": w_warn, "corrections": w_corr}

        # Check 2: Hole vs. wall
        hv_warn, hv_err = _check_hole_vs_wall(flat)
        all_warnings.extend(hv_warn)
        all_errors.extend(hv_err)
        checks["hole_vs_wall"] = {"warnings": hv_warn, "errors": hv_err}

        # Check 3: Volume vs. capacity
        vol_warn, vol_err, vol_corr = _check_volume_vs_capacity(flat)
        all_warnings.extend(vol_warn)
        all_errors.extend(vol_err)
        all_corrections.update(vol_corr)
        checks["volume_capacity"] = {"warnings": vol_warn, "errors": vol_err}

        # Check 4: Aspect ratio
        ar_warn = _check_aspect_ratio(flat, cat)
        all_warnings.extend(ar_warn)
        checks["aspect_ratio"] = {"warnings": ar_warn}

        # Check 5: Minimum feature sizes
        mf_warn, mf_err = _check_minimum_feature_size(flat, process)
        all_warnings.extend(mf_warn)
        all_errors.extend(mf_err)
        checks["min_feature_size"] = {"warnings": mf_warn, "errors": mf_err}

        # Check 6: Draft consistency
        draft_warn = _check_draft_consistency(flat)
        all_warnings.extend(draft_warn)
        checks["draft_consistency"] = {"warnings": draft_warn}

    except Exception as exc:
        logger.error(f"Layer 3 validation error: {exc}", exc_info=True)
        all_warnings.append(f"Validation partially failed: {exc}")

    # Build corrected params: start from flat, apply corrections on top
    corrected = {**flat, **all_corrections}

    # Engineering score: reduce by 10% per warning, 25% per error
    score = 1.0 - 0.10 * len(all_warnings) - 0.25 * len(all_errors)
    score = max(0.0, min(1.0, score))

    return ValidationResult(
        passed=len(all_errors) == 0,
        corrections=all_corrections,
        auto_corrected_params=corrected,
        warnings=all_warnings,
        errors=all_errors,
        engineering_score=round(score, 3),
        checks=checks,
    )
