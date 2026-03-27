"""
Engineering Math Engine — Layer 1 & 3 of the Hybrid AI Accuracy System.

Provides deterministic mathematical formulas, physics laws, manufacturing rules,
industrial dimension databases, and parametric constraint validation for
geometrically precise and industrially accurate 3D model generation.

No AI calls — pure deterministic mathematics.

Implements:
  - Core Geometric Formulas (volume, surface area, aspect ratio, circumference)
  - Golden Ratio & Rule of Thirds (aesthetic proportions)
  - G2 Curvature Continuity (Class A surfacing math)
  - Draft Angle Trigonometric Clearance (injection molding)
  - Hooke's Law (material deformation)
  - Newton's Laws (force & impact)
  - Torque Capacity (rotating parts)
  - Ideal Gas Law (sealed containers)
  - GD&T Tolerance Engine (ISO 286 grades)
  - Bernoulli's Equation (fluid-containing parts)
  - Euler Buckling Formula (slender structures)
  - Thin-Wall Pressure Vessel (Barlow's formula)
  - Heat Dissipation / Fourier's Law (thermal)
  - Pareto Prioritizer (80/20 for feature complexity)
  + Industrial dimension DB (20+ product families)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Physical & mathematical constants ─────────────────────────────────────────

PHI          = (1 + math.sqrt(5)) / 2      # Golden Ratio φ ≈ 1.6180
INV_PHI      = 1 / PHI                      # 1/φ ≈ 0.6180
R_GAS        = 8.314                        # J/(mol·K) — universal gas constant
GRAVITY      = 9806.65                      # mm/s² (= 9.80665 m/s²)
PI           = math.pi

# ── Material modulus of elasticity (MPa) ──────────────────────────────────────
ELASTIC_MODULI: dict[str, float] = {
    "ABS":        2200.0,
    "PLA":        3500.0,
    "PETG":       2100.0,
    "Nylon":      2800.0,
    "TPU":         400.0,
    "Al6061":   68900.0,
    "Steel304": 193000.0,
    "Polycarbonate": 2600.0,
    "Resin":      3000.0,
    "HDPE":       1100.0,
}

# ── Mechanical Standards ───────────────────────────────────────────────────────

# ISO 611 standard bearing sizes (outer diameter, inner diameter, width) in mm
BEARING_STANDARDS: dict[str, dict[str, float]] = {
    "608": {"od": 22.0, "id": 8.0,  "w": 7.0},  # Fidget spinner standard
    "624": {"od": 13.0, "id": 4.0,  "w": 5.0},
    "623": {"od": 10.0, "id": 3.0,  "w": 4.0},
    "688": {"od": 16.0, "id": 8.0,  "w": 5.0},
    "MR105": {"od": 10.0, "id": 5.0, "w": 4.0},
}


def bearing_seat_dim(bearing_name: str = "608", fit_type: str = "press") -> dict[str, float]:
    """
    Return the required bore dimensions for a standard bearing seat.
    
    fit_type:
      - "press": H7 fit (−0.02mm interference)
      - "slip":  G7 fit (+0.03mm clearance)
      - "loose": +0.10mm clearance (for non-functional mockups)
    """
    std = BEARING_STANDARDS.get(bearing_name, BEARING_STANDARDS["608"])
    od = std["od"]
    
    if fit_type == "press":
        bore_r = (od - 0.02) / 2.0
    elif fit_type == "slip":
        bore_r = (od + 0.03) / 2.0
    else:
        bore_r = (od + 0.10) / 2.0
        
    return {
        "bore_radius_mm": round(float(bore_r), 3),  # type: ignore
        "bore_depth_mm":  float(std["w"]),
        "outer_radius_mm": round(float((od / 2.0) + 3.0), 3),  # type: ignore
    }


# ── Geometric Utility Formulas ────────────────────────────────────────────────

def calculate_stack_overlap_z(h_bottom: float, h_top: float, overlap: float = 2.0) -> float:
    """
    Returns the Z-center position for a part being stacked on top of another,
    assuming BOTH are centered at their local origin (CadQuery default).
    
    Rule: pos_z = (h_bottom / 2.0) + (h_top / 2.0) - overlap
    """
    return (h_bottom / 2.0) + (h_top / 2.0) - overlap


# ── Minimum wall thickness by manufacturing process (mm) ─────────────────────
MIN_WALL_THICKNESS: dict[str, float] = {
    "FDM":            1.5,
    "SLA":            0.5,
    "SLS":            0.8,
    "MJF":            0.8,
    "Injection":      1.0,
    "CNC":            0.8,
    "CastUrethane":   1.0,
    "SheetMetal":     0.8,
}

# ── Standard fastener clearance holes (ISO 273 fine fit) (mm) ─────────────────
ISO_CLEARANCE_HOLES: dict[str, float] = {
    "M2": 2.2, "M2.5": 2.7, "M3": 3.4, "M4": 4.5,
    "M5": 5.5, "M6": 6.6, "M8": 9.0, "M10": 11.0, "M12": 13.5,
}

# ISO 4762 Socket Head Cap Screw — counterbore dimensions {bolt_size: (cbore_dia, cbore_depth, head_height)}
ISO_COUNTERBORE: dict[str, tuple[float, float, float]] = {
    "M2":  (4.4,  2.0,  2.0),
    "M2.5":(5.4,  2.5,  2.5),
    "M3":  (6.5,  3.0,  3.0),
    "M4":  (8.25, 4.0,  4.0),
    "M5":  (9.75, 5.0,  5.0),
    "M6":  (11.25,6.0,  6.0),
    "M8":  (14.25,8.0,  8.0),
    "M10": (17.25,10.0, 10.0),
    "M12": (19.75,12.0, 12.0),
}

# ISO 10642 Countersink dimensions {bolt_size: (head_dia, angle_deg)}
ISO_COUNTERSINK: dict[str, tuple[float, float]] = {
    "M2":  (4.4,  90.0),
    "M2.5":(5.5,  90.0),
    "M3":  (6.3,  90.0),
    "M4":  (8.4,  90.0),
    "M5":  (10.4, 90.0),
    "M6":  (12.6, 90.0),
    "M8":  (17.3, 90.0),
    "M10": (20.0, 90.0),
}


def get_hole_dimensions(bolt_size: str, hole_type: str = "through",
                         fit_type: str = "clearance") -> dict:
    """
    Get precise hole dimensions for a given bolt size and hole type.
    Returns dict with keys: diameter, depth (None=through), cbore_dia, cbore_depth, csk_dia, csk_angle.
    """
    result = {"diameter": 0.0, "depth": None}

    # Base hole diameter
    if fit_type == "press":
        # Nominal diameter (tight fit)
        nominal = float(bolt_size.replace("M", ""))
        result["diameter"] = nominal - 0.02
    elif fit_type == "close":
        # Close clearance
        nominal = float(bolt_size.replace("M", ""))
        result["diameter"] = nominal + 0.1
    else:
        # Standard clearance (ISO 273 medium)
        result["diameter"] = ISO_CLEARANCE_HOLES.get(bolt_size, float(bolt_size.replace("M", "")) + 0.5)

    # Hole type additions
    if hole_type == "counterbore" and bolt_size in ISO_COUNTERBORE:
        cbore = ISO_COUNTERBORE[bolt_size]
        result["cbore_dia"] = cbore[0]
        result["cbore_depth"] = cbore[1]
    elif hole_type == "countersink" and bolt_size in ISO_COUNTERSINK:
        csk = ISO_COUNTERSINK[bolt_size]
        result["csk_dia"] = csk[0]
        result["csk_angle"] = csk[1]

    return result

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. CORE GEOMETRIC FORMULAS                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def circle_area(radius_mm: float) -> float:
    """A = π·r²  — cross-section area for material and flow calculations."""
    return PI * radius_mm ** 2


def circumference(radius_mm: float) -> float:
    """C = 2π·r  — critical for machining tolerances and part fitting."""
    return 2 * PI * radius_mm


def cylinder_volume(radius_mm: float, height_mm: float) -> float:
    """V = π·r²·h  — fluid capacity and weight estimate."""
    return circle_area(radius_mm) * height_mm


def cylinder_surface_area(radius_mm: float, height_mm: float, closed: bool = True) -> float:
    """SA = 2π·r·h + 2π·r² (closed) — coating / paint material estimate."""
    lateral = 2 * PI * radius_mm * height_mm
    caps = 2 * PI * radius_mm ** 2 if closed else 0.0
    return lateral + caps


def sphere_volume(radius_mm: float) -> float:
    """V = (4/3)·π·r³"""
    return (4 / 3) * PI * radius_mm ** 3


def sphere_surface_area(radius_mm: float) -> float:
    """SA = 4·π·r² — coating / heat dissipation estimate."""
    return 4 * PI * radius_mm ** 2


def box_volume(l: float, w: float, h: float) -> float:
    """V = l·w·h"""
    return l * w * h


def box_surface_area(l: float, w: float, h: float) -> float:
    """SA = 2(lw + lh + wh)"""
    return 2 * (l * w + l * h + w * h)


def cone_volume(r_base: float, r_top: float, height: float) -> float:
    """V = (π·h/3)·(r_base² + r_base·r_top + r_top²)  — frustum formula."""
    return (PI * height / 3) * (r_base**2 + r_base * r_top + r_top**2)


def torus_volume(major_r: float, minor_r: float) -> float:
    """V = 2π²·R·r² — volume of a torus (donut / ring)."""
    return 2 * PI**2 * major_r * minor_r**2


def pythagorean_diagonal(a: float, b: float) -> float:
    """c = √(a² + b²) — diagonal bracing, slope and internal distance."""
    return math.sqrt(a**2 + b**2)


def aspect_ratio(dimension_a: float, dimension_b: float) -> float:
    """a:b ratio — visual balance and brand consistency check."""
    if dimension_b == 0:
        return float("inf")
    return dimension_a / dimension_b


def hollow_shell_volume(outer_r: float, wall_t: float, height: float) -> float:
    """Volume of hollow cylinder (container internal capacity)."""
    inner_r = max(0.0, outer_r - wall_t)
    return cylinder_volume(inner_r, height)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. GOLDEN RATIO & AESTHETIC PROPORTION LAWS                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def golden_ratio_pair(base: float) -> tuple[float, float]:
    """
    Given a base dimension, return (short, long) where long/short = φ.
    Use to generate balanced height:width proportions.
    """
    return base, base * PHI


def golden_ratio_score(dimension_a: float, dimension_b: float) -> float:
    """
    Returns 0.0–1.0 score of how close the ratio is to φ.
    1.0 = perfect golden ratio.
    """
    ratio = max(dimension_a, dimension_b) / max(min(dimension_a, dimension_b), 1e-9)
    deviation = abs(ratio - PHI) / PHI
    return max(0.0, 1.0 - deviation)


def rule_of_thirds_position(total_length: float, index: int = 1) -> float:
    """
    Returns the 1/3 or 2/3 position along a dimension.
    index=1 → 1/3 point (lower focal), index=2 → 2/3 point (upper focal).
    Used to place buttons, screen bezels, nozzles at visual sweet spots.
    """
    return total_length * (index / 3.0)


def fibonacci_sequence(n: int) -> list[int]:
    """
    Generate Fibonacci sequence up to n terms — used for scaling series
    in modular product families (e.g., size S, M, L, XL containers).
    """
    seq = [1, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return [seq[i] for i in range(min(n, len(seq)))]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. G2 CURVATURE CONTINUITY (CLASS A SURFACING)                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def g2_fillet_radius(adjacent_radius: float, smoothness_factor: float = 1.5) -> float:
    """
    For G2 (curvature) continuity, the transition fillet radius must be at
    least smoothness_factor × the adjacent feature radius.

    G2 condition: κ(s) must be equal at the join →  r_fillet ≥ factor · r_adj
    Used by Apple, BMW, Dyson for Class A surface quality.

    Args:
        adjacent_radius: radius of the neighbouring surface feature (mm)
        smoothness_factor: typically 1.5–2.0 for high-end products
    Returns:
        Minimum fillet radius for G2 continuity (mm)
    """
    return adjacent_radius * smoothness_factor


def min_fillet_from_wall(wall_thickness_mm: float) -> float:
    """
    Internal corners must have r ≥ 0.5 × wall_thickness to avoid
    stress concentration (stress factor Kt → 1.0 as r/t → ∞).
    """
    return wall_thickness_mm * 0.5


def curvature_kappa(radius: float) -> float:
    """κ = 1/r — curvature of a circle. G2 requires κ to be equal at join."""
    if radius <= 0:
        return float("inf")
    return 1.0 / radius


def g2_check(r1: float, r2: float, tolerance: float = 0.05) -> dict:
    """
    Check if two surfaces (radii r1, r2) meet at G2 continuity.
    Returns compliance status and deviation.
    """
    k1, k2 = curvature_kappa(r1), curvature_kappa(r2)
    deviation = abs(k1 - k2) / max(abs(k1), abs(k2), 1e-9)
    return {
        "g2_compliant": deviation <= tolerance,
        "curvature_1": round(float(k1), 6),  # type: ignore
        "curvature_2": round(float(k2), 6),  # type: ignore
        "deviation_pct": round(float(deviation * 100), 2),  # type: ignore
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. DRAFT ANGLE — INJECTION MOLDING CLEARANCE                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def min_draft_angle_deg(height_mm: float, min_clearance_mm: float = 0.5) -> float:
    """
    θ = arctan(clearance / height) — minimum draft angle for mold release.
    Standard rule: 1° per 25mm depth.

    Args:
        height_mm: part height / depth of draw (mm)
        min_clearance_mm: minimum physical clearance for ejection (default 0.5mm)
    Returns:
        Minimum draft angle in degrees
    """
    return math.degrees(math.atan(min_clearance_mm / max(height_mm, 1e-9)))


def draft_taper_radius(base_radius: float, height_mm: float, angle_deg: float = 1.5) -> float:
    """
    For a drafted cylinder: r_top = r_base - h × tan(θ)
    Returns the top radius after applying draft angle taper.
    """
    return base_radius - height_mm * math.tan(math.radians(angle_deg))


def draft_compliance(height_mm: float, current_angle_deg: float) -> dict:
    """
    Check if a wall's draft angle meets the injection molding standard (≥1°).
    """
    required = min_draft_angle_deg(height_mm)
    return {
        "compliant": current_angle_deg >= max(required, 1.0),
        "current_deg": round(float(current_angle_deg), 3),  # type: ignore
        "required_deg": round(float(max(required, 1.0)), 3),  # type: ignore
        "recommended_deg": 1.5,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. HOOKE'S LAW — MATERIAL DEFORMATION                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def hookes_deformation(
    force_N: float, length_mm: float,
    cross_area_mm2: float, material: str = "ABS"
) -> float:
    """
    δ = F·L / (A·E)  — axial deformation under load (Hooke's Law).
    Returns deflection in mm. Warns if material yields.
    """
    E = ELASTIC_MODULI.get(material, 2000.0)
    if cross_area_mm2 <= 0 or E <= 0:
        return 0.0
    return (force_N * length_mm) / (cross_area_mm2 * E)


def max_tensile_stress(force_N: float, area_mm2: float) -> float:
    """σ = F/A — axial stress in MPa."""
    return force_N / max(area_mm2, 1e-9)


def spring_stiffness(E: float, area_mm2: float, length_mm: float) -> float:
    """k = E·A/L — spring constant of a column/pin (N/mm)."""
    return (E * area_mm2) / max(length_mm, 1e-9)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  6. NEWTON'S LAWS — FORCE & IMPACT                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def impact_force(mass_g: float, velocity_mm_s: float, stopping_dist_mm: float = 5.0) -> float:
    """
    F = m·v²/(2·d) — force on part during drop/impact (Newtons).
    Uses energy method: KE = ½mv², absorbed over stopping_dist.
    """
    mass_kg = mass_g / 1000.0
    v_m_s = velocity_mm_s / 1000.0
    d_m = stopping_dist_mm / 1000.0
    return (mass_kg * v_m_s**2) / (2 * max(d_m, 1e-6))


# (Simple part_mass_estimate removed; using the material-aware version from below)


def drop_velocity(height_mm: float) -> float:
    """v = √(2gh) — velocity at impact after free-fall from height_mm."""
    h_m = height_mm / 1000.0
    return math.sqrt(2 * 9.80665 * h_m) * 1000.0  # back to mm/s


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  7. TORQUE — ROTATING PARTS & MECHANISMS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def torque(radius_mm: float, force_N: float) -> float:
    """τ = r × F — torque in N·mm for rotating parts, hinges, knobs."""
    return radius_mm * force_N


def torsional_shear_stress(torque_Nmm: float, radius_mm: float) -> float:
    """τ = T·r / J  — shear stress at outer surface of a solid shaft.
    J (polar moment) = π·r⁴/2 for solid circle."""
    J = PI * radius_mm**4 / 2
    return (torque_Nmm * radius_mm) / max(J, 1e-9)


def shaft_min_radius(torque_Nmm: float, shear_strength_MPa: float = 30.0) -> float:
    """
    Minimum shaft radius to safely carry a given torque without yielding.
    From τ = T·r/J → r_min = (2T / π·τ_allow)^(1/3)
    """
    return ((2 * torque_Nmm) / (PI * shear_strength_MPa)) ** (1.0 / 3.0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  8. IDEAL GAS LAW — SEALED & PRESSURISED CONTAINERS                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def ideal_gas_pressure(
    moles: float, temp_K: float, volume_mm3: float
) -> float:
    """
    P = nRT/V — pressure in Pa.
    Converts mm³ → m³ internally.
    Used for pressurised dispensers, sterilisable bottles,
    vacuum-sealed packaging.
    """
    V_m3 = volume_mm3 * 1e-9
    return (moles * R_GAS * temp_K) / max(V_m3, 1e-30)


def container_wall_thickness_pressure(
    pressure_Pa: float, inner_radius_mm: float,
    material_yield_MPa: float = 30.0, safety_factor: float = 3.0
) -> float:
    """
    Barlow's formula (Thin-Wall Pressure Vessel):
    t = P·r / (σ_yield / SF)

    Returns minimum wall thickness in mm for pressurised containers.
    """
    allowable_stress = (material_yield_MPa * 1e6) / safety_factor  # Pa
    t_m = (pressure_Pa * inner_radius_mm * 1e-3) / allowable_stress
    return t_m * 1000.0  # back to mm


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  9. EULER BUCKLING — SLENDER COLUMNS & THIN WALLS                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def euler_buckling_load(
    E_MPa: float, second_moment_mm4: float,
    length_mm: float, end_condition: float = 1.0
) -> float:
    """
    P_cr = π²·E·I / (K·L)²  — critical buckling load for a column (N).
    end_condition K: 1.0 = pinned-pinned, 0.5 = fixed-fixed, 0.7 = fixed-pinned.
    """
    return (PI**2 * E_MPa * second_moment_mm4) / (end_condition * length_mm)**2


def slenderness_ratio(length_mm: float, radius_of_gyration_mm: float) -> float:
    """λ = L/r — slenderness ratio. >200 → definitely buckling risk."""
    return length_mm / max(radius_of_gyration_mm, 1e-9)


def circular_section_I(radius_mm: float) -> float:
    """I = π·r⁴/4 — second moment of area for solid circular cross-section."""
    return PI * radius_mm**4 / 4


def rectangular_section_I(width_mm: float, height_mm: float) -> float:
    """I = b·h³/12 — second moment of area for rectangle (bending about X)."""
    return (width_mm * height_mm**3) / 12


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 10. BERNOULLI'S EQUATION — FLUID FLOW IN NOZZLES & DISPENSERS               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def bernoulli_velocity(
    pressure_Pa: float, density_kg_m3: float = 1000.0
) -> float:
    """
    v = √(2P/ρ) — exit velocity of fluid through nozzle (m/s).
    Used to design soap/lotion dispenser nozzles and pump apertures.
    """
    return math.sqrt(max(0.0, 2 * pressure_Pa / density_kg_m3))


def nozzle_flow_rate_ml_s(
    nozzle_radius_mm: float, pressure_Pa: float,
    density_kg_m3: float = 1000.0, Cd: float = 0.62
) -> float:
    """
    Q = Cd·A·√(2P/ρ) — volumetric flow rate in ml/s.
    Cd: discharge coefficient (0.6–0.65 for sharp-edged orifice).
    """
    A_m2 = PI * (nozzle_radius_mm * 1e-3) ** 2
    v = bernoulli_velocity(pressure_Pa, density_kg_m3)
    Q_m3_s = Cd * A_m2 * v
    return Q_m3_s * 1e6  # ml/s


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 11. FOURIER'S LAW — HEAT DISSIPATION AREA                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def heat_flux(
    power_W: float, surface_area_mm2: float,
    h_conv: float = 0.010  # W/(mm²·K) natural convection ≈ 10 W/m²K
) -> float:
    """
    q = P / (h·A) — temperature rise above ambient (K) at steady state.
    Used for electronic enclosures: if ΔT > ~40K → add fins/vents.
    """
    return power_W / (h_conv * max(surface_area_mm2, 1e-9))


def required_fin_area(power_W: float, max_delta_T_K: float = 20.0,
                       h_conv: float = 0.010) -> float:
    """Minimum surface area (mm²) to stay below max_delta_T."""
    return power_W / (h_conv * max_delta_T_K)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 12. PARETO PRINCIPLE — FEATURE COMPLEXITY PRIORITISER                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def pareto_critical_features(features: list[dict]) -> dict:
    """
    Implements the 80/20 Rule for design features.

    Input: list of {name, visual_weight} dicts.
    Returns: top 20% of features that contribute ~80% of visual mass.

    Use to identify which CSG operations are the 'identity' of the object.
    This guides the AI to spend precision budget on the most impactful features.
    """
    if not features:
        return {"critical": [], "supporting": [], "pareto_threshold": 0}

    total_weight = sum(f.get("visual_weight", 1) for f in features)
    sorted_features = sorted(features, key=lambda x: x.get("visual_weight", 1), reverse=True)

    cumulative = 0.0
    critical, supporting = [], []
    for feat in sorted_features:
        cumulative += feat.get("visual_weight", 1) / total_weight
        if cumulative <= 0.80:
            critical.append(feat)
        else:
            supporting.append(feat)

    return {
        "critical": critical,
        "supporting": supporting,
        "pareto_threshold": len(critical),
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 13. GD&T TOLERANCE ENGINE (ISO 286)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def iso_tolerance_microns(nominal_mm: float, grade: int = 7) -> float:
    """
    ISO 286 tolerance grade (IT grade) in microns.
    Approximate formula: IT = 10^((grade-1)*0.1) × geometric_mean_microns

    Grades:  IT1=precision, IT7=general engineering, IT14=rough.
    Returns tolerance band in microns.
    """
    # ISO 286 standard table (simplified geometric interpolation)
    if nominal_mm <= 3:       D = math.sqrt(1 * 3)
    elif nominal_mm <= 6:     D = math.sqrt(3 * 6)
    elif nominal_mm <= 10:    D = math.sqrt(6 * 10)
    elif nominal_mm <= 18:    D = math.sqrt(10 * 18)
    elif nominal_mm <= 30:    D = math.sqrt(18 * 30)
    elif nominal_mm <= 50:    D = math.sqrt(30 * 50)
    elif nominal_mm <= 80:    D = math.sqrt(50 * 80)
    elif nominal_mm <= 120:   D = math.sqrt(80 * 120)
    else:                     D = math.sqrt(120 * 180)

    # Tolerance unit i (microns) = 0.45·D^(1/3) + 0.001·D
    i = 0.45 * (D ** (1/3)) + 0.001 * D
    # IT grade multiplier (IT5=7i, IT6=10i, IT7=16i, IT8=25i, IT9=40i, IT10=64i)
    multipliers = {1:1, 2:1.6, 3:2.5, 4:4, 5:7, 6:10, 7:16, 8:25, 9:40, 10:64, 11:100, 12:160, 13:250, 14:400}
    mult = multipliers.get(grade, 16)
    return i * mult


def fit_class(hole_dia_mm: float, shaft_dia_mm: float) -> dict:
    """
    Classify shaft-hole fit as clearance, transition, or interference.
    Returns fit type and clearance/interference value.
    """
    diff = hole_dia_mm - shaft_dia_mm
    tol = iso_tolerance_microns(hole_dia_mm) / 1000.0  # mm
    if diff > tol:
        fit = "clearance"
    elif diff < -tol:
        fit = "interference"
    else:
        fit = "transition"
    return {
        "fit_type": fit,
        "clearance_mm": round(float(diff), 4),  # type: ignore
        "tolerance_band_mm": round(float(2 * tol), 4),  # type: ignore
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 14. WALL THICKNESS VALIDATOR                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def wall_thickness_check(
    wall_mm: float, process: str = "FDM",
    height_mm: float = 0.0
) -> dict:
    """
    Validate wall thickness against manufacturing process minimums.
    Also applies Euler slenderness check for tall thin walls.
    """
    min_t = MIN_WALL_THICKNESS.get(process, 1.5)
    ok = wall_mm >= min_t

    result = {
        "compliant": ok,
        "wall_mm": round(float(wall_mm), 3),  # type: ignore
        "minimum_mm": min_t,
        "process": process,
        "recommendation": None,
    }

    if not ok:
        result["recommendation"] = (
            f"Increase wall to ≥{min_t}mm for {process} manufacturing."
        )
    elif height_mm > 0:
        # Slenderness check: h/t > 10 is a thin-wall situation
        sr = height_mm / wall_mm
        if sr > 20:
            result["recommendation"] = (
                f"Slenderness ratio h/t = {sr:.1f} > 20 — consider ribbing or increasing wall."
            )

    return result


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 15. SURFACE FINISH & SMOOTHNESS RADIUS LOOKUP                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def recommended_fillet(wall_thickness_mm: float, feature_type: str = "internal") -> float:
    """
    Recommended fillet radius based on feature type:
      internal: r ≥ 0.5 × wall_t (stress concentration)
      external: r ≥ 0.25 × wall_t (aesthetics + moldability)
      G2_class_a: r ≥ 1.5 × adjacent_r (Class A reflection quality)
    """
    factors = {"internal": 0.5, "external": 0.25, "g2_class_a": 1.5}
    return wall_thickness_mm * factors.get(feature_type, 0.5)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 16. STRUCTURAL SYMMETRY CHECKER                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def symmetry_error(positions: list, axis: str = "X", tolerance_mm: float = 0.5) -> float:
    """
    Given a list of [x,y,z] positions, compute the maximum asymmetry deviation
    about the specified axis. Returns max error in mm.
    Use to verify rotational or bilateral symmetry of CSG tree.
    """
    if not positions or axis not in ("X", "Y", "Z"):
        return 0.0
    idx = {"X": 0, "Y": 1, "Z": 2}[axis]
    coords = [p[idx] for p in positions if len(p) > idx]
    if not coords:
        return 0.0
    centroid = sum(coords) / len(coords)
    return max(abs(c - centroid) for c in coords)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 17. WEIGHT & CENTRE OF GRAVITY                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def centre_of_gravity_1d(masses: list[float], positions_mm: list[float]) -> float:
    """
    x_cg = Σ(m·x) / Σm — 1D centre of gravity.
    Extend to 3D by calling once per axis.
    """
    total_m = sum(masses)
    if total_m == 0:
        return 0.0
    return sum(m * x for m, x in zip(masses, positions_mm)) / total_m


def product_balance_score(cg_z_mm: float, total_height_mm: float) -> float:
    """
    Score 0–1 for product ergonomics: CoG at 40–50% height = best grip balance.
    """
    ratio = cg_z_mm / max(total_height_mm, 1e-9)
    return max(0.0, 1.0 - abs(ratio - 0.45) / 0.45)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 18. THREAD / FASTENER GEOMETRY                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def metric_thread_minor_diameter(nominal_mm: float, pitch_mm: float) -> float:
    """d_minor = d_nominal - 1.2269 × pitch  (ISO 68-1 metric thread)."""
    return nominal_mm - 1.2269 * pitch_mm


def bolt_tensile_area(nominal_mm: float, pitch_mm: float) -> float:
    """A_s = π/4 × ((d_2 + d_3)/2)² — ISO tensile stress area."""
    d3 = nominal_mm - 1.2269 * pitch_mm
    d2 = nominal_mm - 0.6495 * pitch_mm
    return PI / 4 * ((d2 + d3) / 2) ** 2


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 19. MOHR'S CIRCLE — COMBINED STRESS STATE                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def mohrs_circle(sigma_x: float, sigma_y: float, tau_xy: float) -> dict:
    """
    Compute principal stresses and max shear from a 2D stress state.
    Used to verify parts under combined bending + torsion, or
    multi-directional loading (e.g., dispenser neck: bending + internal pressure).
    """
    centre = (sigma_x + sigma_y) / 2
    R = math.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
    return {
        "sigma_1": round(float(centre + R), 4),  # type: ignore
        "sigma_2": round(float(centre - R), 4),  # type: ignore
        "tau_max": round(float(R), 4),  # type: ignore
        "angle_deg": round(float(math.degrees(0.5 * math.atan2(2 * tau_xy, sigma_x - sigma_y))), 3),  # type: ignore
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 20. FLUID VOLUME / FILL-LEVEL GEOMETRY                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fill_height_cylinder(target_volume_ml: float, inner_radius_mm: float) -> float:
    """
    Given a target fluid volume (ml = cm³), compute fill height in mm
    for a cylindrical container. h = V(mm³) / (π·r²).
    """
    V_mm3 = target_volume_ml * 1000.0
    return V_mm3 / max(circle_area(inner_radius_mm), 1e-9)


def pyramid_volume(base_l: float, base_w: float, height: float) -> float:
    """V = (1/3)·base_area·h — trapezoidal hopper or pyramid features."""
    return (1/3) * base_l * base_w * height


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ INDUSTRIAL DIMENSION DATABASE — 20+ Product Families                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class IndustrialDims:
    """Standard reference dimensions for a product category."""
    category: str
    body_width_mm: float
    body_depth_mm: float
    body_height_mm: float
    wall_thickness_mm: float
    primary_aperture_mm: float  # nozzle / opening / bore
    golden_ratio_target: float  # expected H:W ratio
    capacity_ml: Optional[float] = None
    notes: str = ""


INDUSTRIAL_DB: dict[str, IndustrialDims] = {
    "soap_dispenser": IndustrialDims(
        "Soap/Lotion Dispenser", 70, 50, 180, 2.5, 8.0, 2.57, 300,
        "Pump neck Ø12mm, nozzle Ø6mm, shoulder R=0.4×body_r"
    ),
    "shampoo_bottle": IndustrialDims(
        "Shampoo Bottle", 65, 50, 200, 2.0, 28.0, 3.08, 400,
        "Flip-top cap OD≈28mm, shoulder taper 30°"
    ),
    "water_bottle_500ml": IndustrialDims(
        "Water Bottle 500ml", 70, 70, 190, 2.5, 28.0, 2.71, 500,
        "PCO-1881 neck standard, H:D≈2.7"
    ),
    "spray_bottle": IndustrialDims(
        "Trigger Spray Bottle", 80, 60, 240, 2.5, 24.0, 3.0, 500,
        "Trigger housing adds ~70mm to H, shroud R=12mm"
    ),
    "coffee_mug": IndustrialDims(
        "Coffee Mug", 80, 80, 95, 4.0, 75.0, 1.19, 350,
        "Handle cross-section 15×20mm, finger clearance ≥40mm"
    ),
    "smartphone": IndustrialDims(
        "Smartphone Body", 75, 8, 160, 0.8, 0.0, 2.13,
        notes="Button recesses 0.5mm, camera island ≤3mm proud"
    ),
    "tv_remote": IndustrialDims(
        "TV Remote Control", 45, 18, 170, 2.0, 0.0, 3.78,
        notes="Button dome diameter 8–12mm, grip taper 4°"
    ),
    "flashlight": IndustrialDims(
        "Flashlight / Torch", 35, 35, 140, 3.0, 28.0, 4.0,
        notes="Knurling depth 0.5mm, grip dia Ø30–38mm"
    ),
    "door_knob": IndustrialDims(
        "Door Knob", 60, 60, 70, 3.0, 15.0, 1.17,
        notes="Spindle hole Ø15mm square, rose plate Ø65mm"
    ),
    "gear_knob": IndustrialDims(
        "Gear Shift Knob", 40, 40, 60, 4.0, 12.0, 1.5,
        notes="M12×1.25 thread bore standard, grip spherical R=25mm"
    ),
    "medicine_bottle": IndustrialDims(
        "Medicine/Pill Bottle", 45, 45, 90, 2.0, 32.0, 2.0, 100,
        "Child-resistant cap OD≈32mm, label area ≥60% of body"
    ),
    "perfume_bottle": IndustrialDims(
        "Perfume Bottle", 55, 30, 120, 3.0, 15.0, 2.18, 100,
        "Crimped collar Ø15mm, shoulder fillet R≥8mm (Class A)"
    ),
    "toothbrush": IndustrialDims(
        "Toothbrush Handle", 14, 9, 180, 2.5, 0.0, 12.86,
        notes="Head width 10mm, bristle block 35×12mm"
    ),
    "pen": IndustrialDims(
        "Ballpoint Pen", 10, 10, 145, 1.5, 1.0, 14.5,
        notes="Clip width 6mm, ink cartridge OD=3.9mm"
    ),
    "usb_stick": IndustrialDims(
        "USB Flash Drive", 22, 10, 58, 1.5, 12.0, 2.64,
        notes="USB-A plug 12×4.5mm, cap friction fit ±0.1mm"
    ),
    "bracket_L": IndustrialDims(
        "L-Bracket", 50, 4, 50, 4.0, 4.5, 1.0,
        notes="M4 clearance holes 4.5mm, corner fillet R=3mm"
    ),
    "pcb_enclosure": IndustrialDims(
        "PCB Enclosure", 100, 60, 35, 2.5, 0.0, 1.67,
        notes="Snap-fit lug 1×3mm, boss dia=6mm, counterbore Ø3.5mm"
    ),
    "battery_cover": IndustrialDims(
        "Battery Cover (AA×2)", 60, 32, 12, 1.5, 0.0, 0.2,
        notes="Living hinge t=0.5mm, snap-hook 2×1mm"
    ),
    "gear_spur": IndustrialDims(
        "Spur Gear (mod 1)", 30, 8, 8, 5.0, 6.0, 1.0,
        notes="Pressure angle 20°, root fillet r=0.38m, addendum a=m"
    ),
    "pipe_fitting": IndustrialDims(
        "Pipe Elbow Fitting", 40, 40, 60, 3.5, 20.0, 1.5,
        notes="BSP/NPT thread, flow bend radius ≥1.5×ID"
    ),
}


def lookup_reference_dims(object_name: str) -> Optional[IndustrialDims]:
    """
    Fuzzy lookup for industrial standard dimensions.
    Returns the closest match or None.
    """
    key = object_name.lower().replace(" ", "_").replace("-", "_")
    if key in INDUSTRIAL_DB:
        return INDUSTRIAL_DB[key]

    # Partial match
    for db_key, dims in INDUSTRIAL_DB.items():
        tokens = set(db_key.split("_"))
        query_tokens = set(key.split("_"))
        if query_tokens & tokens:
            return dims

    return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FULL ENGINEERING REPORT — runs all checks on a CSG tree result               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class EngineeringReport:
    object_name: str
    volume_mm3: float = 0.0
    surface_area_mm2: float = 0.0
    mass_g: float = 0.0
    golden_ratio_score: float = 0.0
    aspect_ratio_hw: float = 0.0
    wall_compliance: dict = field(default_factory=dict)
    draft_compliance: dict = field(default_factory=dict)
    g2_notes: list = field(default_factory=list)
    reference_dims: Optional[IndustrialDims] = None
    dimension_accuracy_pct: float = 0.0
    alerts: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def generate_engineering_report(
    object_name: str,
    width_mm: float, depth_mm: float, height_mm: float,
    wall_mm: float = 2.5,
    process: str = "FDM",
    primary_radius_mm: float = 0.0,
    material: str = "ABS",
) -> EngineeringReport:
    """
    Run the full engineering accuracy check on a generated object.
    Returns an EngineeringReport with scores, flags, and recommendations.
    """
    report = EngineeringReport(object_name=object_name)

    # Volume & mass (approximated as hollow cylinder if radius given)
    if primary_radius_mm > 0:
        report.volume_mm3 = cylinder_volume(primary_radius_mm, height_mm)
        report.surface_area_mm2 = cylinder_surface_area(primary_radius_mm, height_mm)
    else:
        report.volume_mm3 = box_volume(width_mm, depth_mm, height_mm)
        report.surface_area_mm2 = box_surface_area(width_mm, depth_mm, height_mm)

    density = {"ABS": 1.05, "PLA": 1.24, "PETG": 1.23, "Al6061": 2.70, "Steel304": 7.85}.get(material, 1.1)
    report.mass_g = part_mass_estimate(report.volume_mm3, density)

    # Golden ratio
    report.golden_ratio_score = golden_ratio_score(height_mm, width_mm)
    report.aspect_ratio_hw = aspect_ratio(height_mm, width_mm)

    # Wall thickness compliance
    report.wall_compliance = wall_thickness_check(wall_mm, process, height_mm)

    # Draft angle check (assume 1.5° standard)
    report.draft_compliance = draft_compliance(height_mm, 1.5)

    # G2 fillet recommendation
    g2_r = g2_fillet_radius(primary_radius_mm if primary_radius_mm > 0 else wall_mm * 2)
    report.g2_notes = [
        f"Recommended shoulder fillet: ≥{g2_r:.1f}mm for G2 curvature continuity",
        f"Internal corner fillet: ≥{min_fillet_from_wall(wall_mm):.1f}mm to prevent stress concentration",
    ]

    # Reference dimensions lookup
    ref = lookup_reference_dims(object_name)
    report.reference_dims = ref
    if ref:
        w_err = abs(width_mm  - ref.body_width_mm)  / ref.body_width_mm
        h_err = abs(height_mm - ref.body_height_mm) / ref.body_height_mm
        report.dimension_accuracy_pct = round((1 - (w_err + h_err) / 2) * 100, 1)
        if report.dimension_accuracy_pct < 80:
            report.alerts.append(
                f"Dimensions deviate {100 - report.dimension_accuracy_pct:.1f}% from "
                f"industry standard ({ref.body_width_mm}×{ref.body_height_mm}mm)."
            )

    # Golden ratio alert
    if report.golden_ratio_score < 0.7:
        report.recommendations.append(
            f"Adjust H:W to ≈{PHI:.2f}:1 (currently {report.aspect_ratio_hw:.2f}:1) "
            f"for golden ratio proportions."
        )

    # Wall compliance alerts
    if not report.wall_compliance.get("compliant"):
        report.alerts.append(report.wall_compliance.get("recommendation", "Wall too thin."))

    return report



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE SUPPORT FUNCTIONS — used by 8-layer pipeline services             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def pareto_critical_features(
    features: list,
    weight_key: str = "visual_weight",
    threshold: float = 0.80,
) -> dict:
    """
    Pareto (80/20) sort of features by a numeric weight field.

    Args:
        features: list of dicts each containing at least {name, weight_key}
        weight_key: dict key holding the numeric priority weight
        threshold: cumulative weight fraction for 'critical' bucket (default 80%)
    Returns:
        {"critical": [...], "supporting": [...]}  — each preserves original dict
    """
    if not features:
        return {"critical": [], "supporting": []}

    sorted_f = sorted(features, key=lambda x: float(x.get(weight_key, 1.0)), reverse=True)
    total = sum(float(f.get(weight_key, 1.0)) for f in sorted_f)
    if total <= 0:
        return {"critical": sorted_f, "supporting": []}

    cumulative = 0.0
    critical, supporting = [], []
    total_f = float(total) if total > 0 else 1e-9
    for f in sorted_f:
        cumulative += float(f.get(weight_key, 1.0))
        if cumulative / total_f <= threshold:
            critical.append(f)
        else:
            supporting.append(f)

    if not critical and sorted_f:
        critical.append(sorted_f[0])
        supporting = [sorted_f[i] for i in range(1, len(sorted_f))]

    return {"critical": critical, "supporting": supporting}


def fill_height_cylinder(capacity_ml: float, radius_mm: float, fill_fraction: float = 1.0) -> float:
    """
    Inverse of cylinder_volume: given capacity and radius, return fill height (mm).
    """
    volume_mm3 = capacity_ml * 1000.0
    if radius_mm <= 0:
        return 0.0
    full_height = volume_mm3 / (PI * radius_mm ** 2)
    return full_height / max(fill_fraction, 1e-6)


def part_mass_estimate(volume_mm3: float, material_or_density: str | float = "ABS") -> float:
    """
    Estimate part mass in grams from volume (mm³) and material density.
    """
    if isinstance(material_or_density, (float, int)):
        density_g_cm3 = float(material_or_density)
    else:
        DENSITIES: dict = {
            "ABS": 1.04, "PLA": 1.24, "PETG": 1.27, "Nylon": 1.14,
            "TPU": 1.20, "Polycarbonate": 1.20, "Resin": 1.12,
            "HDPE": 0.95, "Al6061": 2.70, "Steel304": 7.93,
        }
        density_g_cm3 = float(DENSITIES.get(material_or_density, 1.04))
    
    density_g_mm3 = density_g_cm3 / 1000.0
    return volume_mm3 * density_g_mm3


def iso_tolerance_microns(nominal_mm: float, grade: int = 7) -> float:
    """
    Approximate ISO 286 IT tolerance in micrometres for a given nominal size and grade.
    """
    IT_FACTORS: dict = {
        1: 2.5, 2: 4, 3: 6, 4: 10, 5: 16, 6: 25,
        7: 40, 8: 64, 9: 100, 10: 160, 11: 250, 12: 400,
    }
    D = max(nominal_mm, 1.0)
    i = 0.45 * D ** (1.0 / 3.0) + 0.001 * D
    factor = IT_FACTORS.get(grade, 40)
    return round(float(i * factor), 2)  # type: ignore


def wall_thickness_check(wall_mm: float, process: str = "FDM", height_mm: float = 0.0) -> dict:
    """
    Check wall thickness against manufacturing minimums.
    Returns a compliance dict with recommendation if failing.
    """
    process_key = process.title() if process else "FDM"
    min_t = MIN_WALL_THICKNESS.get(process_key, MIN_WALL_THICKNESS.get(process.upper(), 1.5))
    compliant = wall_mm >= min_t
    result: dict = {"compliant": compliant, "wall_mm": wall_mm, "min_mm": min_t, "process": process_key}
    if not compliant:
        result["recommendation"] = (
            f"Wall {wall_mm:.2f}mm < minimum {min_t}mm for {process_key}. Increase to ≥{min_t}mm."
        )
    if height_mm > 0 and wall_mm > 0:
        sr = height_mm / wall_mm
        if sr > 20:
            result["slenderness_warning"] = f"Slenderness h/t={sr:.1f} > 20. Add ribs."
    return result


def report_to_dict(report: "EngineeringReport") -> dict:

    """Convert an EngineeringReport to a JSON-serializable dict."""
    ref = report.reference_dims
    return {
        "object_name": report.object_name,
        "volume_mm3": round(float(report.volume_mm3), 1),  # type: ignore
        "volume_ml": round(float(report.volume_mm3 / 1000), 2),  # type: ignore
        "surface_area_mm2": round(float(report.surface_area_mm2), 1),  # type: ignore
        "mass_g": round(float(report.mass_g), 2),  # type: ignore
        "golden_ratio_score": round(float(report.golden_ratio_score), 3),  # type: ignore
        "aspect_ratio_hw": round(float(report.aspect_ratio_hw), 3),  # type: ignore
        "phi": round(float(PHI), 4),  # type: ignore
        "wall_compliance": report.wall_compliance,
        "draft_compliance": report.draft_compliance,
        "g2_notes": report.g2_notes,
        "dimension_accuracy_pct": round(float(report.dimension_accuracy_pct), 2),  # type: ignore
        "reference": {
            "category": ref.category,
            "width_mm": ref.body_width_mm,
            "height_mm": ref.body_height_mm,
            "wall_mm": ref.wall_thickness_mm,
            "aperture_mm": ref.primary_aperture_mm,
            "capacity_ml": ref.capacity_ml,
            "notes": ref.notes,
        } if ref else None,
        "alerts": report.alerts,
        "recommendations": report.recommendations,
    }
