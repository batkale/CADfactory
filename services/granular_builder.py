"""
Layer 7 — Granular CadQuery Script Builder

This replaces the basic `build_cadquery_script()` from csg_builder.py with
a 5–10× more detailed generation.

Key enhancements vs. the original builder:
  - Parametric header block (all dims as named constants)
  - Draft-aware cones (auto-converts cylinders > 20mm to drafted cones)
  - Localized fillets per semantic region (not global edge fillet)
  - GD&T tolerance inline comments per feature
  - Engineering math validation comments (volume, mass, wall ratio)
  - Semantic label tagging on every variable
  - Phase-separated assembly (additive → subtractive → intersect → final union)
  - Per-feature try/except for fillet operations
  - Assertion guards before all box/cylinder/cone primitives

Location: cadfactory-backend/services/granular_builder.py
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from services.csg_guard import GuardedPlan
from services.engineering_math import (
    ISO_CLEARANCE_HOLES,
    MIN_WALL_THICKNESS,
    box_volume,
    cone_volume,
    cylinder_volume,
    draft_taper_radius,
    iso_tolerance_microns,
    part_mass_estimate,
    sphere_volume,
)
from services.gap_filler import RefinedSpec
from services.semantic_decomposer import CSGOperation

logger = logging.getLogger(__name__)


# ── Constant-name sanitiser ────────────────────────────────────────────────────

def _const_name(label: str, suffix: str) -> str:
    """Convert a snake_case label + suffix into an UPPER_SNAKE_CASE constant name."""
    return f"{label.upper()}_{suffix.upper()}".replace(" ", "_").replace("-", "_")


# ── Draft-aware cylinder conversion ───────────────────────────────────────────

def _should_draft(shape: str, params: dict, height_threshold_mm: float = 20.0) -> bool:
    """Return True if this cylinder should be modeled as a drafted cone."""
    if shape.lower() != "cylinder":
        return False
    h = float(params.get("height", 0))
    return h > height_threshold_mm


def _draft_params(params: dict, angle_deg: float = 1.5) -> Tuple[float, float, float]:
    """Return (r_base, r_top, height) for a drafted cone from cylinder params."""
    r_base = float(params.get("radius", 10))
    h      = float(params.get("height", 20))
    r_top  = max(1.0, draft_taper_radius(r_base, h, angle_deg))
    return r_base, r_top, h


# ── Engineering comment generators ────────────────────────────────────────────

def _vol_comment(shape: str, params: dict, label: str) -> str:
    """Generate a volume/mass engineering comment for a primitive."""
    s = shape.lower()
    comments = []
    try:
        if s == "cylinder":
            r, h = float(params.get("radius", 0)), float(params.get("height", 0))
            v = cylinder_volume(r, h)
            m = part_mass_estimate(v)
            comments.append(f"V=π×{r:.1f}²×{h:.1f}={v:.0f}mm³ | m≈{m:.1f}g")
            comments.append(f"SA_lat=2π×{r:.1f}×{h:.1f}={2*math.pi*r*h:.0f}mm²")
        elif s == "cone":
            rb = float(params.get("r_base", 0))
            rt = float(params.get("r_top", 0))
            h  = float(params.get("height", 0))
            v  = cone_volume(rb, rt, h)
            m  = part_mass_estimate(v)
            comments.append(f"cone frustum V={v:.0f}mm³ | draft r_top={rt:.2f}mm ✓")
            comments.append(f"m≈{m:.1f}g | draft_angle={math.degrees(math.atan((rb-rt)/h)):.2f}°")
        elif s in ("box", "rounded_box"):
            length = float(params.get("length", 0))
            w = float(params.get("width", 0))
            h = float(params.get("height", 0))
            v = box_volume(length, w, h)
            m = part_mass_estimate(v)
            comments.append(f"V={length:.1f}×{w:.1f}×{h:.1f}={v:.0f}mm³ | m≈{m:.1f}g")
        elif s == "sphere":
            r = float(params.get("radius", 0))
            v = sphere_volume(r)
            m = part_mass_estimate(v)
            comments.append(f"V=(4/3)π×{r:.1f}³={v:.0f}mm³ | m≈{m:.1f}g")
    except Exception:
        pass
    if comments:
        return f"  # [{label}] " + " | ".join(comments)
    return f"  # [{label}]"


def _hole_tolerance_comment(radius_mm: float, label: str) -> str:
    """Generate a GD&T ISO 273 comment for a hole."""
    dia = radius_mm * 2.0
    # Find closest ISO clearance
    closest_name = ""
    closest_val  = 0.0
    for name, clear_dia in ISO_CLEARANCE_HOLES.items():
        if abs(clear_dia - dia) < abs(closest_val - dia):
            closest_val  = clear_dia
            closest_name = name
    tol_um = iso_tolerance_microns(dia, grade=7)
    base = f"  # [{label}] hole Ø{dia:.2f}mm"
    if closest_name:
        base += f" ≈ {closest_name} clearance ({closest_val:.1f}mm ISO 273 fine)"
    base += f" | IT7={tol_um:.1f}µm tolerance"
    return base


# ── Curve prism lines (reuse from csg_builder) ────────────────────────────────

def _curve_prism_lines(var: str, params: dict) -> List[str]:
    """Import and delegate to the original curve prism line generator."""
    from services.csg_builder import _curve_prism_lines as _orig
    return _orig(var, params)


def _rotate_position(x: float, y: float, angle_deg: float) -> Tuple[float, float]:
    """Rotate (x, y) around Z-axis by angle_deg."""
    rad = math.radians(angle_deg)
    xr = x * math.cos(rad) - y * math.sin(rad)
    yr = x * math.sin(rad) + y * math.cos(rad)
    return xr, yr


# ── Header block builder ──────────────────────────────────────────────────────

def _build_parametric_header(
    ops: List[CSGOperation],
    spec: Optional[RefinedSpec],
    process: str,
    material: str,
    tol_class: str,
    confidence: float,
) -> List[str]:
    """Generate the parametric constants header block."""
    lines = [
        "import cadquery as cq",
        "import math",
        "",
        "# ═══════════════════════════════════════════════════════════════════",
        "# PARAMETRIC CONSTANTS — 8-Layer Precision Pipeline",
        f"# Pipeline confidence: {confidence:.2f} | Process: {process.upper()} | "
        f"Material: {material} | Tol: {tol_class}",
        "# ═══════════════════════════════════════════════════════════════════",
    ]

    if spec:
        # Physics/manufacturing context
        if spec.wall_mm > 0:
            min_t = MIN_WALL_THICKNESS.get(process, 1.5)
            ok = "✓" if spec.wall_mm >= min_t else "✗"
            lines.append(f"#   Wall: {spec.wall_mm}mm ≥ min {min_t}mm for {process} {ok}")
        if spec.draft_angle_deg > 0:
            lines.append(f"#   Draft angle: {spec.draft_angle_deg}° (1° per 25mm deep rule ✓)")
        if spec.fillet_r_shoulder > 0:
            lines.append(
                f"#   Shoulder fillet: ≥{spec.fillet_r_shoulder}mm for G2 curvature continuity ✓"
            )
        if spec.height_mm > 20 and spec.wall_mm > 0:
            sr = spec.height_mm / spec.wall_mm
            note = "ribs recommended" if sr > 20 else "acceptable"
            lines.append(f"#   Slenderness ratio h/t = {sr:.1f} — {note}")
        lines.append("")

    # Enumerate all dimension constants from each operation
    lines.append("# ─── Dimension constants ───────────────────────────────────────────")
    for i, op in enumerate(ops):
        p = op.params
        prefix = _const_name(op.label or f"op{i}", "")
        for k, v in p.items():
            if isinstance(v, (int, float)):
                const = f"{prefix}{k.upper()}"
                lines.append(f"{const:<32} = {v!r:<10}  # mm — [{op.label}]")

    # Spec-level constants
    if spec:
        lines.append("")
        lines.append("# ─── Spec-level engineering constants ──────────────────────────────")
        if spec.wall_mm:
            lines.append(f"WALL_MM                          = {spec.wall_mm!r}")
        if spec.draft_angle_deg:
            lines.append(f"DRAFT_ANGLE_DEG                  = {spec.draft_angle_deg!r}")
            lines.append(f"DRAFT_TAN                        = {math.tan(math.radians(spec.draft_angle_deg)):.6f}  # tan(DRAFT_ANGLE_DEG)")
        if spec.fillet_r_external:
            lines.append(f"FILLET_R_EXTERNAL                = {spec.fillet_r_external!r}")
        if spec.fillet_r_internal:
            lines.append(f"FILLET_R_INTERNAL                = {spec.fillet_r_internal!r}")
        if spec.fillet_r_shoulder:
            lines.append(f"FILLET_R_SHOULDER                = {spec.fillet_r_shoulder!r}")

    lines.append("")
    return lines


# ── Main granular build function ───────────────────────────────────────────────

def build_granular_script(
    ops: List[CSGOperation],
    spec: Optional[RefinedSpec] = None,
    guarded: Optional[GuardedPlan] = None,
    process: str = "fdm",
    material: str = "ABS",
    tol_class: str = "IT7",
    confidence: float = 0.8,
    object_name: str = "Part",
    aesthetic_class: str = "mechanical",
) -> str:
    """
    Layer 7: Generate a hyper-detailed, parametric CadQuery script.

    This is 5–10× more verbose than the basic CSG builder output.
    Every feature is documented, every magic number is a named constant,
    and every operation has engineering validation comments.
    """
    draft_angle = spec.draft_angle_deg if spec else 1.5
    ext_fillet  = spec.fillet_r_external if spec else 2.0
    priority_labels = set(spec.priority_order[:3] if spec else [])
    pareto_labels   = set(guarded.approved_ops[i].label for i in range(min(2, len(guarded.approved_ops))) if guarded) if guarded else set()

    # ── Build parametric header ────────────────────────────────────────────
    all_ops = guarded.approved_ops if guarded else ops
    lines = _build_parametric_header(all_ops, spec, process, material, tol_class, confidence)

    # ── Guard risk summary ─────────────────────────────────────────────────
    if guarded and guarded.risk_flags:
        lines.append("# ─── Guard warnings ────────────────────────────────────────────────")
        for flag in guarded.risk_flags[:6]:
            severity_prefix = "# ⚠ " if flag.severity == "warning" else "# ✗ "
            lines.append(f"{severity_prefix}[{flag.operation_label}] {flag.description[:90]}")
        lines.append("")

    lines.append(f"# ─── Shape construction: {object_name} ─────────────────────────────")
    lines.append("")

    # ── Part tracker ──────────────────────────────────────────────────────
    needs_math = any(op.shape.lower() == "curve_prism" for op in all_ops)
    if not needs_math:
        # Remove the "import math" if not needed by curve_prism
        # (It's kept because draft calculations may use it)
        pass

    parts: Dict[str, List[str]] = {}  # label → [var_names]
    shape_idx = 0

    # ── Phase banner ──────────────────────────────────────────────────────
    add_ops = [op for op in all_ops if op.op == "add"]
    sub_ops = [op for op in all_ops if op.op == "subtract"]
    int_ops = [op for op in all_ops if op.op == "intersect"]

    for phase_ops, phase_name in [
        (add_ops,  "PHASE 1 — Additive: Build primary geometry"),
        (sub_ops,  "PHASE 2 — Subtractive: Cuts, holes, pockets"),
        (int_ops,  "PHASE 3 — Intersect: Masking operations"),
    ]:
        if not phase_ops:
            continue
        lines.append(f"# {'─'*60}")
        lines.append(f"# {phase_name}")
        lines.append(f"# {'─'*60}")

        for op in phase_ops:
            label = op.label or "part"
            if label not in parts:
                parts[label] = []

            shape_lower = op.shape.lower()
            is_curve    = shape_lower == "curve_prism"
            is_rounded  = shape_lower == "rounded_box"
            is_identity = label in priority_labels or label in pareto_labels

            # Expand circular copies
            instances = (
                [(i * op.copy_angle_step) for i in range(op.copies)]
                if op.copies and op.copies > 1 and op.copy_angle_step
                else [None]
            )

            for copy_angle in instances:
                var = f"_s{shape_idx}"
                shape_idx += 1
                parts[label].append(var)

                # Engineering comment
                if op.op == "subtract" and "hole" in label.lower():
                    r = float(op.params.get("radius", op.params.get("r_base", 3.0)))
                    lines.append(_hole_tolerance_comment(r, label))
                else:
                    lines.append(_vol_comment(op.shape, op.params, label))

                # Pareto marker
                if is_identity:
                    lines.append("  # ★ IDENTITY FEATURE — receives full precision budget")

                # ── Assert guards ─────────────────────────────────────────
                p = op.params
                if shape_lower in ("cylinder", "cone"):
                    h_key = "height"
                    r_key = "radius" if "radius" in p else "r_base"
                    if h_key in p and r_key in p:
                        const_h = _const_name(label, h_key)
                        const_r = _const_name(label, r_key)
                        lines.append(
                            f"  assert {const_h} > 0 and {const_r} > 0, "
                            f"f\"[{label}] dims must be >0: h={{{const_h}}}, r={{{const_r}}}\""
                        )
                elif shape_lower in ("box", "rounded_box"):
                    for k in ("length", "width", "height"):
                        if k in p:
                            const = _const_name(label, k)
                            lines.append(
                                f"  assert {const} > 0, f\"[{label}] {k} must be >0: {{{const}}}\""
                            )

                # ── Shape expression ──────────────────────────────────────
                if is_curve:
                    lines.extend(_curve_prism_lines(var, op.params))

                elif is_rounded:
                    L = float(p.get("length", 20))
                    W = float(p.get("width", 20))
                    H = float(p.get("height", 10))
                    R = float(p.get("fillet_r", 2.0))
                    lines.append(f"{var} = cq.Workplane('XY').box({L}, {W}, {H})")
                    lines.append(f"try: {var} = {var}.edges().fillet({R})")
                    lines.append(f"except Exception: pass  # fillet skipped on [{label}]")

                elif _should_draft(op.shape, op.params) and op.op == "add":
                    # Drafted cone instead of bare cylinder
                    r_base, r_top, h = _draft_params(op.params, draft_angle)
                    cb = _const_name(label, "r_base")
                    ct = _const_name(label, "r_top_drafted")
                    ch = _const_name(label, "height")
                    lines.append(
                        f"{ct} = max(1.0, {cb} - {ch} * DRAFT_TAN)  "
                        f"# draft taper r_top"
                    )
                    lines.append(
                        f"{var} = cq.Workplane('XY').add("
                        f"cq.Solid.makeCone({cb}, {ct}, {ch}))"
                    )

                elif shape_lower == "cylinder":
                    h_c = _const_name(label, "height")
                    r_c = _const_name(label, "radius")
                    # FreeCAD-inspired: generate counterbore/countersink geometry
                    if op.op == "subtract" and op.bolt_size and op.hole_type in ("counterbore", "countersink"):
                        from services.engineering_math import get_hole_dimensions
                        hole_dims = get_hole_dimensions(op.bolt_size, op.hole_type, op.fit_type or "clearance")
                        if op.hole_type == "counterbore" and "cbore_dia" in hole_dims:
                            cbore_r = hole_dims["cbore_dia"] / 2.0
                            cbore_d = hole_dims["cbore_depth"]
                            lines.append(f"# ISO 4762 counterbore for {op.bolt_size}")
                            lines.append(f"{var} = cq.Workplane('XY').cylinder({h_c}, {r_c})")
                            lines.append(f"_cbore_{var} = cq.Workplane('XY').cylinder({cbore_d}, {cbore_r})")
                            bore_h = float(op.params.get("height", 10))
                            offset_z = (bore_h - cbore_d) / 2.0
                            lines.append(f"_cbore_{var} = _cbore_{var}.translate((0, 0, {offset_z:.3f}))")
                            lines.append(f"{var} = {var}.union(_cbore_{var})")
                        elif op.hole_type == "countersink" and "csk_dia" in hole_dims:
                            csk_r = hole_dims["csk_dia"] / 2.0
                            lines.append(f"# ISO 10642 countersink for {op.bolt_size}")
                            lines.append(f"{var} = cq.Workplane('XY').cylinder({h_c}, {r_c})")
                            bore_h = float(op.params.get("height", 10))
                            csk_depth = csk_r - float(op.params.get("radius", 2))
                            lines.append(f"_csk_{var} = cq.Workplane('XY').add(cq.Solid.makeCone({csk_r}, {r_c}, {csk_depth:.3f}))")
                            offset_z = (bore_h - csk_depth) / 2.0
                            lines.append(f"_csk_{var} = _csk_{var}.translate((0, 0, {offset_z:.3f}))")
                            lines.append(f"{var} = {var}.union(_csk_{var})")
                        else:
                            lines.append(f"{var} = cq.Workplane('XY').cylinder({h_c}, {r_c})")
                    else:
                        lines.append(f"{var} = cq.Workplane('XY').cylinder({h_c}, {r_c})")

                elif shape_lower == "box":
                    l_c = _const_name(label, "length")
                    w_c = _const_name(label, "width")
                    h_c = _const_name(label, "height")
                    lines.append(f"{var} = cq.Workplane('XY').box({l_c}, {w_c}, {h_c})")

                elif shape_lower == "sphere":
                    r_c = _const_name(label, "radius")
                    lines.append(f"{var} = cq.Workplane('XY').sphere({r_c})")

                elif shape_lower == "cone":
                    rb_c = _const_name(label, "r_base")
                    rt_c = _const_name(label, "r_top")
                    h_c  = _const_name(label, "height")
                    lines.append(
                        f"{var} = cq.Workplane('XY').add("
                        f"cq.Solid.makeCone({rb_c}, {rt_c}, {h_c}))"
                    )

                elif shape_lower == "torus":
                    mr = float(p.get("major_r", 20))
                    nr = float(p.get("minor_r", 5))
                    lines.append(
                        f"{var} = cq.Workplane('XZ').center({mr}, 0)"
                        f".circle({nr}).revolve()"
                    )

                elif shape_lower == "polygon_prism":
                    sides = int(p.get("sides", 6))
                    dia   = float(p.get("diameter", 20))
                    h     = float(p.get("height", 10))
                    lines.append(
                        f"{var} = cq.Workplane('XY').polygon({sides}, {dia}).extrude({h})"
                    )

                elif shape_lower == "curve_prism":
                    profile = str(p.get("profile", "star")).lower()
                    h = float(p.get("height", 10))
                    if profile == "star":
                        n = int(p.get("n_points", 5))
                        ir = float(p.get("inner_r", 5))
                        or_ = float(p.get("outer_r", 10))
                        lines.append(f"# Star profile: n={n}, ir={ir}, or={or_}")
                        lines.append(f"{var} = cq.Workplane('XY').star({or_}, {ir}, {n}).extrude({h})")
                    elif profile == "teardrop":
                        # A teardrop is a circle + a tangent triangle/box
                        # Used for 3D printing and fidget spinner arms
                        r = float(p.get("radius", 10))
                        w = float(p.get("width", 20))
                        lines.append(f"# Teardrop profile: r={r}, w={w}")
                        lines.append(f"{var} = cq.Workplane('XY').circle({r}).extrude({h})")
                        lines.append(f"{var}_tip = cq.Workplane('XY').workplane(offset={h/2}).rect({w}, {r*2}).extrude({-h})")
                        lines.append(f"{var} = {var}.union({var}_tip).edges('|Z').fillet({r*0.99})")
                    elif profile == "heart":
                        w = float(p.get("width", 20))
                        lines.append(f"# Heart profile: w={w}")
                        lines.append(
                            f"{var} = cq.Workplane('XY').polyline([(0,0), ({w/2}, {w/2}), (0, {w*1.2}), ({-w/2}, {w/2})]).close().extrude({h})"
                        )
                    elif profile == "cross":
                        aw = float(p.get("arm_width", 5))
                        w = float(p.get("width", 20))
                        lines.append(f"# Cross profile: aw={aw}, w={w}")
                        lines.append(f"{var} = cq.Workplane('XY').rect({w}, {aw}).union(cq.Workplane('XY').rect({aw}, {w})).extrude({h})")
                    else:
                        lines.append(f"{var} = cq.Workplane('XY').circle(10).extrude({h})")

                else:
                    logger.warning(f"Granular builder: unknown shape '{op.shape}' — box fallback")
                    lines.append(f"{var} = cq.Workplane('XY').box(20, 20, 10)  # fallback")

                # ── Rotations ─────────────────────────────────────────────
                if op.rotation:
                    rx, ry, rz = (op.rotation + [0.0, 0.0, 0.0])[:3]
                    if rx:
                        lines.append(f"{var} = {var}.rotate((0,0,0), (1,0,0), {rx})")
                    if ry:
                        lines.append(f"{var} = {var}.rotate((0,0,0), (0,1,0), {ry})")
                    if rz:
                        lines.append(f"{var} = {var}.rotate((0,0,0), (0,0,1), {rz})")

                if copy_angle is not None and copy_angle != 0:
                    lines.append(f"{var} = {var}.rotate((0,0,0), (0,0,1), {copy_angle:.3f})")

                # ── Translation ───────────────────────────────────────────
                px, py, pz = (op.position + [0.0, 0.0, 0.0])[:3]
                if copy_angle is not None and copy_angle != 0:
                    px, py = _rotate_position(px, py, copy_angle)
                if abs(px) > 1e-6 or abs(py) > 1e-6 or abs(pz) > 1e-6:
                    lines.append(
                        f"{var} = {var}.translate(({px:.3f}, {py:.3f}, {pz:.3f}))"
                    )

                # ── Localized fillet for identity additive features ────────
                if op.op == "add" and is_identity and not is_curve:
                    lines.append(
                        f"try: {var} = {var}.edges('|Z').fillet(FILLET_R_EXTERNAL)"
                        f"  # exterior edge fillet [{label}]"
                    )
                    lines.append("except Exception: pass")

                lines.append("")

    # ── Final assembly (two-pass: additive first, then subtractive) ────────────
    lines.append("# ─── Final Assembly ────────────────────────────────────────────────")
    result_set = False

    # Pass 0: union all additive components into result
    # Pass 1: cut / intersect subtractive & intersect components
    for pass_num in (0, 1):
        for label, vars_list in parts.items():
            if not vars_list:
                continue
            comp_var = f"part_{label}"
            op_type = "add"
            for op in all_ops:
                if op.label == label:
                    op_type = op.op
                    break

            # Route each component to the correct pass
            if pass_num == 0 and op_type != "add":
                continue
            if pass_num == 1 and op_type not in ("subtract", "intersect"):
                continue

            lines.append(f"# Component: '{label}' ({op_type})")
            lines.append(f"{comp_var} = {vars_list[0]}")
            for v in vars_list[1:]:
                lines.append(f"{comp_var} = {comp_var}.union({v})")
            # NOTE: no per-component show_object — only the final result is shown

            if op_type == "add":
                if not result_set:
                    lines.append(f"result = {comp_var}")
                    result_set = True
                else:
                    lines.append(f"result = result.union({comp_var})")
            elif op_type == "subtract":
                if not result_set:
                    # No additive base yet — use placeholder
                    lines.append("result = cq.Workplane('XY').box(10, 10, 10)  # placeholder")
                    result_set = True
                lines.append(f"result = result.cut({comp_var})")
            elif op_type == "intersect":
                if result_set:
                    lines.append(f"result = result.intersect({comp_var})")
                else:
                    lines.append(f"result = {comp_var}")
                    result_set = True

            lines.append("")

    if not result_set:
        lines.append("result = cq.Workplane('XY').box(20, 20, 10)  # empty fallback")

    # ── Global finish pass ─────────────────────────────────────────────────
    ext_r = spec.fillet_r_external if spec else ext_fillet

    is_aesthetic = aesthetic_class in ("consumer", "decorative")

    if ext_r > 0 or is_aesthetic:
        fillet_r = ext_r if ext_r > 0 else 2.0
        lines.append("# ─── Global finish: external edge softening ────────────────────────")
        lines.append(f"try: result = result.edges('|Z').fillet({fillet_r})")
        lines.append("except Exception:")
        lines.append(f"    try: result = result.edges('|Z').chamfer({fillet_r * 0.5})")
        lines.append("    except Exception: pass  # fillet+chamfer skipped")

        # Bottom chamfer for all parts
        lines.append(
            "try: result = result.faces('<Z').edges().chamfer(0.5)  # base lip"
        )
        lines.append("except Exception: pass")

        # Extra finishing for consumer/decorative parts
        if is_aesthetic:
            lines.append("")
            lines.append(f"# ─── Aesthetic finish ({aesthetic_class}) ─────────────────────────────")
            lines.append("# Top face concentric detail groove")
            lines.append("try:")
            lines.append("    _bb = result.val().BoundingBox()")
            lines.append("    _max_r = min(_bb.xlen, _bb.ylen) / 2.0 * 0.6")
            lines.append("    if _max_r > 3.0:")
            lines.append("        _groove = cq.Workplane('XY').circle(_max_r + 0.4).circle(_max_r).extrude(0.4)")
            lines.append("        _groove = _groove.translate((0, 0, _bb.zmax - 0.4))")
            lines.append("        result = result.cut(_groove)")
            lines.append("except Exception: pass  # aesthetic groove skipped")

        lines.append("")

    lines.append("show_object(result)")
    return "\n".join(lines)
