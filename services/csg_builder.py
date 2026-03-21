"""
CSG Builder — deterministic conversion of a DecomposedObject into a CadQuery script.

No AI used — pure geometry math and code generation.

Location: cadfactory-backend/services/csg_builder.py
"""

from __future__ import annotations

import math
import logging
from typing import List

from services.semantic_decomposer import DecomposedObject, CSGOperation

logger = logging.getLogger(__name__)

MAX_BBOX_MM = 500.0


# ── Curve library ──────────────────────────────────────────────────────────────
# Each entry is a list of code-line templates that build `{pts_var}` as a list
# of (x, y, 0.0) tuples (open polygon — close step added by caller).
# Templates may use: n_pts, and shape-specific variables extracted from params.

def _curve_prism_lines(var: str, params: dict) -> List[str]:
    """
    Return a list of Python source lines that define `var` as a
    cq.Workplane containing a curve_prism solid.

    Supported curve names (params["curve"]):
      heart     — parametric valentine heart
      star      — N-pointed star  (n_points, outer_r, inner_r)
      teardrop  — water-drop / teardrop (width, depth)
      leaf      — lens / leaf shape     (width, depth)
      diamond   — elongated 4-point diamond (width, depth)
      arrow     — arrowhead pointing up (width, depth, shaft_width)
      cross     — plus / cross shape    (width, depth, arm_width)
    """
    curve   = str(params.get("curve", "heart")).lower()
    height  = float(params.get("height", 5.0))
    N       = 100                     # sample points for parametric curves
    pv      = f"_pts{var}"           # e.g. _pts_s0

    lines: List[str] = []

    # ── parametric curves (need a loop) ───────────────────────────────────────
    if curve == "heart":
        s = float(params.get("scale", params.get("width", 32.0) / 32.0))
        lines += [
            f"{pv} = []",
            f"for _i in range({N}):",
            f"    _t = 2.0 * _math.pi * _i / {N}",
            f"    _x = float(16.0 * _math.sin(_t)**3 * {s:.4f})",
            f"    _y = float((13.0*_math.cos(_t) - 5.0*_math.cos(2*_t)"
            f" - 2.0*_math.cos(3*_t) - _math.cos(4*_t)) * {s:.4f})",
            f"    {pv}.append((_x, _y, 0.0))",
        ]

    elif curve == "star":
        n_pts_star = max(3, int(float(params.get("n_points", 5))))
        outer_r = float(params.get("outer_r", 20.0))
        inner_r = float(params.get("inner_r", outer_r * 0.4))
        n_total = 2 * n_pts_star
        lines += [
            f"{pv} = []",
            f"for _i in range({n_total}):",
            f"    _ang = _math.pi * _i / {n_pts_star} - _math.pi / 2",
            f"    _r = {outer_r:.3f} if _i % 2 == 0 else {inner_r:.3f}",
            f"    {pv}.append((_r * _math.cos(_ang), _r * _math.sin(_ang), 0.0))",
        ]

    elif curve == "teardrop":
        # Bottom is pointed (t=0), top is rounded (t=π)
        w2 = float(params.get("width", 20.0)) / 2.0
        sy = float(params.get("depth", params.get("height_2d", 30.0))) / 2.0
        lines += [
            f"{pv} = []",
            f"for _i in range({N}):",
            f"    _t = 2.0 * _math.pi * _i / {N}",
            f"    _x = float(_math.sin(_t) * {w2:.3f})",
            f"    _y = float((-_math.cos(_t) + 0.5*_math.cos(_t)**2 - 0.5) * {sy:.3f})",
            f"    {pv}.append((_x, _y, 0.0))",
        ]

    elif curve == "leaf":
        # Lens shape — more pointed than an ellipse (tips at ±width/2, y=0)
        w2 = float(params.get("width", 25.0)) / 2.0
        h2 = float(params.get("depth", params.get("height_2d", 40.0))) / 2.0
        lines += [
            f"{pv} = []",
            f"for _i in range({N}):",
            f"    _t = 2.0 * _math.pi * _i / {N}",
            f"    _st = _math.sin(_t)",
            f"    _x = float(_math.cos(_t) * {w2:.3f})",
            f"    _y = float(_math.copysign(abs(_st)**1.5, _st) * {h2:.3f})",
            f"    {pv}.append((_x, _y, 0.0))",
        ]

    # ── polygon shapes (hardcoded points) ─────────────────────────────────────
    elif curve == "diamond":
        w2 = float(params.get("width", 20.0)) / 2.0
        h2 = float(params.get("depth", params.get("height_2d", 30.0))) / 2.0
        lines += [
            f"{pv} = [({w2:.3f},0.0,0.0),(0.0,{h2:.3f},0.0),"
            f"(-{w2:.3f},0.0,0.0),(0.0,-{h2:.3f},0.0)]",
        ]

    elif curve == "arrow":
        w   = float(params.get("width", 30.0))
        h   = float(params.get("depth", params.get("height_2d", 40.0)))
        sw  = float(params.get("shaft_width", w * 0.35)) / 2.0
        hw  = w / 2.0;  hh = h / 2.0
        lines += [
            f"{pv} = [(0.0,{hh:.3f},0.0),({hw:.3f},0.0,0.0),({sw:.3f},0.0,0.0),"
            f"({sw:.3f},-{hh:.3f},0.0),(-{sw:.3f},-{hh:.3f},0.0),"
            f"(-{sw:.3f},0.0,0.0),(-{hw:.3f},0.0,0.0)]",
        ]

    elif curve == "cross":
        w   = float(params.get("width",  30.0))
        h   = float(params.get("depth", params.get("height_2d", 30.0)))
        aw  = float(params.get("arm_width", min(w, h) * 0.35)) / 2.0
        hw  = w / 2.0;  hh = h / 2.0
        lines += [
            f"{pv} = [({aw:.3f},{hh:.3f},0.0),(-{aw:.3f},{hh:.3f},0.0),"
            f"(-{aw:.3f},{aw:.3f},0.0),(-{hw:.3f},{aw:.3f},0.0),"
            f"(-{hw:.3f},-{aw:.3f},0.0),(-{aw:.3f},-{aw:.3f},0.0),"
            f"(-{aw:.3f},-{hh:.3f},0.0),({aw:.3f},-{hh:.3f},0.0),"
            f"({aw:.3f},-{aw:.3f},0.0),({hw:.3f},-{aw:.3f},0.0),"
            f"({hw:.3f},{aw:.3f},0.0),({aw:.3f},{aw:.3f},0.0)]",
        ]

    else:
        # Unknown curve — fall back to circle approximation
        r = float(params.get("width", params.get("radius", 20.0))) / 2.0
        logger.warning(f"Unknown curve '{curve}' — falling back to circle, r={r}")
        lines += [
            f"{pv} = []",
            f"for _i in range({N}):",
            f"    _t = 2.0 * _math.pi * _i / {N}",
            f"    {pv}.append((_math.cos(_t) * {r:.3f}, _math.sin(_t) * {r:.3f}, 0.0))",
        ]

    # Close polygon + build Wire → Face → Solid → Workplane
    lines += [
        f"{pv}.append({pv}[0])",
        f"_wire{var} = cq.Wire.makePolygon({pv})",
        f"_face{var} = cq.Face.makeFromWires(_wire{var})",
        f"_solid{var} = cq.Solid.extrudeLinear(_face{var}, cq.Vector(0.0, 0.0, {height:.3f}))",
        f"{var} = cq.Workplane('XY').add(_solid{var})",
    ]
    return lines


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_csg_tree(obj: DecomposedObject) -> List[str]:
    """
    Check the CSG tree for physically nonsensical values.
    Returns a list of error strings (empty = OK).
    """
    errors: List[str] = []

    for i, op in enumerate(obj.operations):
        p = op.params
        shape = op.shape.lower()
        prefix = f"Operation {i} ({shape})"

        # Generic: all numeric params must be positive
        # Exceptions: r_top=0 is valid (sharp cone tip); fillet_r can be tiny
        _zero_allowed = {"r_top", "fillet_r"}
        for k, v in p.items():
            if isinstance(v, (int, float)) and k not in ("sides", *_zero_allowed):
                if float(v) <= 0:
                    errors.append(f"{prefix}: param '{k}' must be > 0, got {v}")

        # Shape-specific checks
        if shape == "cylinder":
            if "radius" not in p:
                errors.append(f"{prefix}: missing 'radius'")
            if "height" not in p:
                errors.append(f"{prefix}: missing 'height'")

        elif shape == "box":
            for key in ("length", "width", "height"):
                if key not in p:
                    errors.append(f"{prefix}: missing '{key}'")

        elif shape == "sphere":
            if "radius" not in p:
                errors.append(f"{prefix}: missing 'radius'")

        elif shape == "polygon_prism":
            sides = p.get("sides", 0)
            if not isinstance(sides, (int, float)) or int(sides) < 3:
                errors.append(f"{prefix}: 'sides' must be >= 3, got {sides}")
            if "diameter" not in p:
                errors.append(f"{prefix}: missing 'diameter'")
            if "height" not in p:
                errors.append(f"{prefix}: missing 'height'")

        elif shape == "curve_prism":
            if "curve" not in p:
                errors.append(f"{prefix}: missing 'curve' (e.g. 'heart', 'star', 'teardrop')")
            if "height" not in p:
                errors.append(f"{prefix}: missing 'height'")
            valid_curves = {"heart", "star", "teardrop", "leaf", "diamond", "arrow", "cross"}
            if p.get("curve", "").lower() not in valid_curves:
                errors.append(
                    f"{prefix}: unknown curve '{p.get('curve')}'. "
                    f"Valid: {', '.join(sorted(valid_curves))}"
                )

        # Copy angle check
        if op.copies and op.copy_angle_step:
            total_angle = op.copies * op.copy_angle_step
            if total_angle > 360.0 + 1e-3:
                errors.append(
                    f"{prefix}: copies={op.copies} × angle_step={op.copy_angle_step}°"
                    f" = {total_angle}° exceeds 360°"
                )

    # Bounding box warning (not an error)
    dims = obj.estimated_dimensions
    for key, val in dims.items():
        if isinstance(val, (int, float)) and float(val) > MAX_BBOX_MM:
            errors.append(
                f"WARNING: estimated dimension '{key}' = {val} mm exceeds {MAX_BBOX_MM} mm"
            )

    return errors


# ── Code generation ────────────────────────────────────────────────────────────

def _build_primitive_expr(shape: str, params: dict) -> str:
    """Return a CadQuery expression string for a single primitive (no translate/rotate)."""
    p = params
    s = shape.lower()

    if s == "cylinder":
        h = float(p.get("height", 20))
        r = float(p.get("radius", 10))
        return f"cq.Workplane('XY').cylinder({h}, {r})"

    elif s == "box":
        l = float(p.get("length", 20))
        w = float(p.get("width", 20))
        h = float(p.get("height", 10))
        return f"cq.Workplane('XY').box({l}, {w}, {h})"

    elif s == "sphere":
        r = float(p.get("radius", 10))
        return f"cq.Workplane('XY').sphere({r})"

    elif s == "cone":
        h = float(p.get("height", 20))
        r_base = float(p.get("r_base", p.get("radius", 10)))
        r_top = float(p.get("r_top", 0))
        return f"cq.Workplane('XY').add(cq.Solid.makeCone({r_base}, {r_top}, {h}))"

    elif s == "torus":
        major_r = float(p.get("major_r", 20))
        minor_r = float(p.get("minor_r", 5))
        return f"cq.Workplane('XZ').center({major_r}, 0).circle({minor_r}).revolve()"

    elif s == "rounded_box":
        l = float(p.get("length", 20))
        w = float(p.get("width", 20))
        h = float(p.get("height", 10))
        r = float(p.get("fillet_r", min(l, w, h) * 0.1))
        # Fillet wrapped in try/except to avoid CadQuery topology failures
        return (
            f"_tmp_box_cq.Workplane('XY').box({l}, {w}, {h})"
        ).replace("_tmp_box_", "")  # placeholder — handled below

    elif s == "polygon_prism":
        sides = int(p.get("sides", 6))
        diameter = float(p.get("diameter", 20))
        height = float(p.get("height", 10))
        return f"cq.Workplane('XY').polygon({sides}, {diameter}).extrude({height})"

    else:
        # Fallback: box
        logger.warning(f"Unknown shape '{shape}', falling back to box")
        return "cq.Workplane('XY').box(20, 20, 10)"


def _rotate_position(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    """Rotate (x, y) around Z by angle_deg."""
    rad = math.radians(angle_deg)
    xr = x * math.cos(rad) - y * math.sin(rad)
    yr = x * math.sin(rad) + y * math.cos(rad)
    return xr, yr


def build_cadquery_script(obj: DecomposedObject) -> str:
    """
    Convert a DecomposedObject into a runnable CadQuery Python script.

    Generates one shape variable per primitive (including per copy),
    then applies CSG operations sequentially.
    """
    needs_math = any(op.shape.lower() == "curve_prism" for op in obj.operations)
    lines: List[str] = [
        "import cadquery as cq",
        *(["import math as _math"] if needs_math else []),
        f"# {obj.object_name} — generated by CSG builder",
        f"# real-world reference: {obj.real_world_reference}",
        f"# dims estimated: {obj.dims_are_estimated}",
        "",
    ]

    result_set = False
    shape_idx = 0

    # Track rounded_box shapes that need multi-line fillet handling
    rounded_vars: List[str] = []

    for op in obj.operations:
        shape_lower = op.shape.lower()
        is_rounded = shape_lower == "rounded_box"

        # Expand copies into individual instances
        if op.copies and op.copies > 1 and op.copy_angle_step:
            instances = [
                (i * op.copy_angle_step) for i in range(op.copies)
            ]
        else:
            instances = [None]

        for copy_angle in instances:
            var = f"_s{shape_idx}"
            shape_idx += 1

            is_curve = shape_lower == "curve_prism"

            # Build base expression / multi-line block
            if is_curve:
                # Multi-line: generates `var = cq.Workplane(...)`
                lines.extend(_curve_prism_lines(var, op.params))
            elif is_rounded:
                p = op.params
                l = float(p.get("length", 20))
                w = float(p.get("width", 20))
                h = float(p.get("height", 10))
                r = float(p.get("fillet_r", min(l, w, h) * 0.1))
                prim = f"cq.Workplane('XY').box({l}, {w}, {h})"
                lines.append(f"{var} = {prim}")
            else:
                prim = _build_primitive_expr(op.shape, op.params)
                lines.append(f"{var} = {prim}")

            # Apply shape-level rotation (from op.rotation) — as separate statement
            # (works for both single-expr and multi-line curve_prism)
            if op.rotation:
                rx, ry, rz = (op.rotation + [0.0, 0.0, 0.0])[:3]
                if rx:
                    lines.append(f"{var} = {var}.rotate((0,0,0), (1,0,0), {rx})")
                if ry:
                    lines.append(f"{var} = {var}.rotate((0,0,0), (0,1,0), {ry})")
                if rz:
                    lines.append(f"{var} = {var}.rotate((0,0,0), (0,0,1), {rz})")

            # Apply Z-rotation for copy symmetry
            if copy_angle is not None and copy_angle != 0:
                lines.append(f"{var} = {var}.rotate((0,0,0), (0,0,1), {copy_angle:.3f})")

            # Compute translated position (rotate position vector for copies)
            px, py, pz = (op.position + [0.0, 0.0, 0.0])[:3]
            if copy_angle is not None and copy_angle != 0:
                px, py = _rotate_position(px, py, copy_angle)

            if abs(px) > 1e-6 or abs(py) > 1e-6 or abs(pz) > 1e-6:
                lines.append(f"{var} = {var}.translate(({px:.3f}, {py:.3f}, {pz:.3f}))")

            # For rounded_box, add fillet in try/except on next lines
            if is_rounded:
                r = float(op.params.get("fillet_r", 2.0))
                lines.append(f"try:")
                lines.append(f"    {var} = {var}.edges().fillet({r})")
                lines.append(f"except Exception:")
                lines.append(f"    pass  # fillet failed, keeping sharp edges")
                rounded_vars.append(var)

            # Apply CSG operation
            if op.op == "add":
                if not result_set:
                    lines.append(f"result = {var}")
                    result_set = True
                else:
                    lines.append(f"result = result.union({var})")
            elif op.op == "subtract":
                if result_set:
                    lines.append(f"result = result.cut({var})")
                else:
                    logger.warning("Subtract before any add operation — skipping")
            elif op.op == "intersect":
                if result_set:
                    lines.append(f"result = result.intersect({var})")
                else:
                    logger.warning("Intersect before any add operation — skipping")

    # Safety fallback
    if not result_set:
        lines.append("# Fallback: no valid add operations found")
        lines.append("result = cq.Workplane('XY').box(50, 50, 20)")

    lines.append("")
    lines.append("show_object(result)")

    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_from_prompt_result(obj: DecomposedObject) -> dict:
    """
    Validate and build a CadQuery script from a DecomposedObject.

    Returns:
        {"success": True, "errors": [], "script": "..."}
      or
        {"success": False, "errors": [...], "script": None}
    """
    errors = validate_csg_tree(obj)

    # Hard errors (not warnings) block generation
    hard_errors = [e for e in errors if not e.startswith("WARNING:")]
    if hard_errors:
        return {"success": False, "errors": hard_errors, "script": None}

    try:
        script = build_cadquery_script(obj)
    except Exception as e:
        return {"success": False, "errors": [f"Script generation failed: {e}"], "script": None}

    return {"success": True, "errors": errors, "script": script}


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from services.semantic_decomposer import CSGOperation, DecomposedObject

    # Manually construct a fidget spinner
    spinner = DecomposedObject(
        object_name="Fidget Spinner",
        real_world_reference="fidget spinner",
        symmetry="3-fold rotational",
        estimated_dimensions={"overall_diameter_mm": 70, "height_mm": 8},
        dims_are_estimated=True,
        operations=[
            # Central hub
            CSGOperation(op="add", shape="cylinder",
                         params={"radius": 12, "height": 8}, position=[0, 0, 0]),
            # Central bearing hole
            CSGOperation(op="subtract", shape="cylinder",
                         params={"radius": 7, "height": 8}, position=[0, 0, 0]),
            # Three arms
            CSGOperation(op="add", shape="cylinder",
                         params={"radius": 8, "height": 8}, position=[25, 0, 0],
                         copies=3, copy_angle_step=120.0),
            # Bearing holes in arm tips
            CSGOperation(op="subtract", shape="cylinder",
                         params={"radius": 5, "height": 8}, position=[25, 0, 0],
                         copies=3, copy_angle_step=120.0),
        ],
        modifiers=["triangular_arm_tips"],
        confidence=0.9,
    )

    result = build_from_prompt_result(spinner)
    print("Success:", result["success"])
    print("Errors:", result["errors"])
    print("\nGenerated script:")
    print(result["script"])
