"""
Unit tests for engineering_math.py — pure deterministic math functions.

Tests round-trip properties, known values, and edge cases.
Inspired by CadQuery's test suite discipline and GenCAD's lack thereof.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engineering_math import (
    aspect_ratio,
    # Fastener
    bearing_seat_dim,
    # Fluid
    bernoulli_velocity,
    box_surface_area,
    box_volume,
    # Geometric formulas
    circle_area,
    circular_section_I,
    circumference,
    cone_volume,
    container_wall_thickness_pressure,
    curvature_kappa,
    cylinder_surface_area,
    cylinder_volume,
    draft_compliance,
    draft_taper_radius,
    drop_velocity,
    # Buckling
    euler_buckling_load,
    fibonacci_sequence,
    g2_check,
    # G2 curvature
    g2_fillet_radius,
    get_hole_dimensions,
    # Golden ratio
    golden_ratio_pair,
    golden_ratio_score,
    # Thermal
    heat_flux,
    hollow_shell_volume,
    # Hooke's law
    hookes_deformation,
    # Gas law
    ideal_gas_pressure,
    # Newton's laws
    impact_force,
    max_tensile_stress,
    # Draft angle
    min_draft_angle_deg,
    min_fillet_from_wall,
    nozzle_flow_rate_ml_s,
    # Pareto
    pareto_critical_features,
    pythagorean_diagonal,
    rectangular_section_I,
    required_fin_area,
    rule_of_thirds_position,
    shaft_min_radius,
    slenderness_ratio,
    sphere_surface_area,
    sphere_volume,
    spring_stiffness,
    # Torque
    torque,
    torsional_shear_stress,
    torus_volume,
)

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2


# ── Geometric Formulas ───────────────────────────────────────────────────────

class TestGeometricFormulas:
    def test_circle_area_unit(self):
        assert circle_area(1.0) == pytest.approx(PI, rel=1e-10)

    def test_circle_area_known(self):
        assert circle_area(10.0) == pytest.approx(PI * 100, rel=1e-10)

    def test_circumference_unit(self):
        assert circumference(1.0) == pytest.approx(2 * PI, rel=1e-10)

    def test_cylinder_volume(self):
        # r=5, h=10 -> V = pi*25*10 = 250*pi
        assert cylinder_volume(5.0, 10.0) == pytest.approx(250 * PI, rel=1e-10)

    def test_cylinder_surface_area_closed(self):
        # r=1, h=1 -> lateral=2*pi + caps=2*pi = 4*pi
        assert cylinder_surface_area(1.0, 1.0, closed=True) == pytest.approx(4 * PI, rel=1e-10)

    def test_cylinder_surface_area_open(self):
        assert cylinder_surface_area(1.0, 1.0, closed=False) == pytest.approx(2 * PI, rel=1e-10)

    def test_sphere_volume(self):
        # r=1 -> V = 4/3 * pi
        assert sphere_volume(1.0) == pytest.approx(4 * PI / 3, rel=1e-10)

    def test_sphere_surface_area(self):
        assert sphere_surface_area(1.0) == pytest.approx(4 * PI, rel=1e-10)

    def test_box_volume(self):
        assert box_volume(2, 3, 4) == 24

    def test_box_surface_area(self):
        assert box_surface_area(2, 3, 4) == 52

    def test_cone_volume_full_cone(self):
        # r_base=1, r_top=0, h=3 -> V = pi*3/3*(1+0+0) = pi
        assert cone_volume(1.0, 0.0, 3.0) == pytest.approx(PI, rel=1e-10)

    def test_cone_volume_cylinder(self):
        # When r_base == r_top, it's a cylinder: V = pi*r^2*h
        assert cone_volume(2.0, 2.0, 5.0) == pytest.approx(PI * 4 * 5, rel=1e-10)

    def test_torus_volume(self):
        # V = 2*pi^2*R*r^2
        assert torus_volume(10.0, 2.0) == pytest.approx(2 * PI**2 * 10 * 4, rel=1e-10)

    def test_pythagorean_3_4_5(self):
        assert pythagorean_diagonal(3, 4) == pytest.approx(5.0, rel=1e-10)

    def test_aspect_ratio_square(self):
        assert aspect_ratio(10, 10) == pytest.approx(1.0)

    def test_aspect_ratio_zero_division(self):
        assert aspect_ratio(10, 0) == float("inf")

    def test_hollow_shell_volume(self):
        # outer_r=10, wall=2, h=20 -> inner_r=8 -> V=pi*64*20
        assert hollow_shell_volume(10, 2, 20) == pytest.approx(PI * 64 * 20, rel=1e-10)


# ── Golden Ratio & Aesthetics ────────────────────────────────────────────────

class TestGoldenRatio:
    def test_golden_ratio_pair(self):
        short, long = golden_ratio_pair(10.0)
        assert short == 10.0
        assert long == pytest.approx(10.0 * PHI, rel=1e-10)

    def test_golden_ratio_score_perfect(self):
        assert golden_ratio_score(1.0, PHI) == pytest.approx(1.0, abs=0.01)

    def test_golden_ratio_score_bad(self):
        score = golden_ratio_score(1.0, 5.0)
        assert score < 0.5

    def test_rule_of_thirds(self):
        assert rule_of_thirds_position(90, 1) == pytest.approx(30.0)
        assert rule_of_thirds_position(90, 2) == pytest.approx(60.0)

    def test_fibonacci_sequence(self):
        seq = fibonacci_sequence(6)
        assert seq == [1, 1, 2, 3, 5, 8]


# ── G2 Curvature ─────────────────────────────────────────────────────────────

class TestG2Curvature:
    def test_g2_fillet_radius(self):
        assert g2_fillet_radius(2.0, 1.5) == 3.0

    def test_min_fillet_from_wall(self):
        assert min_fillet_from_wall(4.0) == 2.0

    def test_curvature_kappa(self):
        assert curvature_kappa(10.0) == pytest.approx(0.1)
        assert curvature_kappa(0.0) == float("inf")

    def test_g2_check_equal(self):
        result = g2_check(10.0, 10.0)
        assert result["g2_compliant"] is True
        assert result["deviation_pct"] == 0.0

    def test_g2_check_different(self):
        result = g2_check(10.0, 100.0)
        assert result["g2_compliant"] is False


# ── Draft Angle ──────────────────────────────────────────────────────────────

class TestDraftAngle:
    def test_min_draft_angle_short_part(self):
        angle = min_draft_angle_deg(10.0, 0.5)
        assert angle > 0
        assert angle == pytest.approx(math.degrees(math.atan(0.5 / 10.0)), rel=1e-6)

    def test_draft_taper_radius(self):
        r_top = draft_taper_radius(10.0, 20.0, 1.5)
        expected = 10.0 - 20.0 * math.tan(math.radians(1.5))
        assert r_top == pytest.approx(expected, rel=1e-6)

    def test_draft_compliance_good(self):
        result = draft_compliance(25.0, 2.0)
        assert result["compliant"] is True

    def test_draft_compliance_bad(self):
        result = draft_compliance(100.0, 0.1)
        assert result["compliant"] is False


# ── Hooke's Law ──────────────────────────────────────────────────────────────

class TestHookesLaw:
    def test_hookes_deformation_known(self):
        # F=100N, L=100mm, A=10mm^2, E=2000 MPa
        # delta = 100*100 / (10*2000) = 0.5mm
        delta = hookes_deformation(100.0, 100.0, 10.0, "TPU")
        # TPU E=400 -> delta = 100*100/(10*400) = 2.5
        assert delta == pytest.approx(2.5, rel=1e-6)

    def test_hookes_zero_area(self):
        assert hookes_deformation(100, 100, 0, "ABS") == 0.0

    def test_max_tensile_stress(self):
        assert max_tensile_stress(1000, 100) == pytest.approx(10.0)

    def test_spring_stiffness(self):
        # k = E*A/L = 2000*10/100 = 200
        assert spring_stiffness(2000, 10, 100) == pytest.approx(200.0)


# ── Newton's Laws ────────────────────────────────────────────────────────────

class TestNewtonLaws:
    def test_drop_velocity(self):
        # v = sqrt(2*9.80665*1) * 1000 mm/s for 1000mm drop
        v = drop_velocity(1000)
        assert v > 0
        assert v == pytest.approx(math.sqrt(2 * 9.80665 * 1) * 1000, rel=1e-6)

    def test_impact_force_positive(self):
        f = impact_force(100, 1000, 5)
        assert f > 0


# ── Torque ───────────────────────────────────────────────────────────────────

class TestTorque:
    def test_torque_basic(self):
        assert torque(10.0, 5.0) == 50.0

    def test_shaft_min_radius_positive(self):
        r = shaft_min_radius(100.0, 30.0)
        assert r > 0

    def test_torsional_shear_stress(self):
        stress = torsional_shear_stress(100.0, 5.0)
        assert stress > 0


# ── Ideal Gas Law ────────────────────────────────────────────────────────────

class TestGasLaw:
    def test_ideal_gas_pressure_known(self):
        # 1 mol at 300K in 1 m^3 = 1e9 mm^3 -> P = 8.314*300/1 = 2494.2 Pa
        p = ideal_gas_pressure(1.0, 300.0, 1e9)
        assert p == pytest.approx(8.314 * 300, rel=1e-4)

    def test_wall_thickness_positive(self):
        t = container_wall_thickness_pressure(100000, 50.0, 30.0, 3.0)
        assert t > 0


# ── Euler Buckling ───────────────────────────────────────────────────────────

class TestBuckling:
    def test_euler_buckling_known(self):
        # P_cr = pi^2 * E * I / (K*L)^2
        modulus, moi, length, eff_k = 200000, 100, 500, 1.0
        expected = PI**2 * modulus * moi / (eff_k * length)**2
        assert euler_buckling_load(modulus, moi, length, eff_k) == pytest.approx(expected, rel=1e-10)

    def test_slenderness_ratio(self):
        assert slenderness_ratio(1000, 5) == pytest.approx(200.0)

    def test_circular_section_I(self):
        assert circular_section_I(10.0) == pytest.approx(PI * 10**4 / 4, rel=1e-10)

    def test_rectangular_section_I(self):
        assert rectangular_section_I(20, 30) == pytest.approx(20 * 30**3 / 12, rel=1e-10)


# ── Fluid Flow ───────────────────────────────────────────────────────────────

class TestFluidFlow:
    def test_bernoulli_velocity(self):
        # v = sqrt(2*P/rho) = sqrt(2*200000/1000) = sqrt(400) = 20 m/s
        assert bernoulli_velocity(200000, 1000) == pytest.approx(20.0, rel=1e-6)

    def test_bernoulli_zero_pressure(self):
        assert bernoulli_velocity(0, 1000) == 0.0

    def test_nozzle_flow_positive(self):
        q = nozzle_flow_rate_ml_s(2.0, 100000)
        assert q > 0


# ── Thermal ──────────────────────────────────────────────────────────────────

class TestThermal:
    def test_heat_flux_positive(self):
        dt = heat_flux(10.0, 1000.0)
        assert dt > 0

    def test_required_fin_area(self):
        area = required_fin_area(10.0, 20.0, 0.010)
        assert area == pytest.approx(10.0 / (0.010 * 20.0), rel=1e-6)


# ── Pareto ───────────────────────────────────────────────────────────────────

class TestPareto:
    def test_pareto_empty(self):
        result = pareto_critical_features([])
        assert result["critical"] == []
        assert result["supporting"] == []

    def test_pareto_identifies_critical(self):
        features = [
            {"name": "body", "visual_weight": 80},
            {"name": "fillet1", "visual_weight": 5},
            {"name": "fillet2", "visual_weight": 5},
            {"name": "hole1", "visual_weight": 5},
            {"name": "hole2", "visual_weight": 5},
        ]
        result = pareto_critical_features(features)
        assert len(result["critical"]) >= 1
        assert result["critical"][0]["name"] == "body"


# ── Fastener & Bearing ───────────────────────────────────────────────────────

class TestFastenerBearing:
    def test_bearing_seat_press(self):
        dims = bearing_seat_dim("608", "press")
        assert dims["bore_radius_mm"] == pytest.approx((22.0 - 0.02) / 2, rel=1e-3)
        assert dims["bore_depth_mm"] == 7.0

    def test_bearing_seat_slip(self):
        dims = bearing_seat_dim("608", "slip")
        assert dims["bore_radius_mm"] == pytest.approx((22.0 + 0.03) / 2, rel=1e-3)

    def test_hole_dimensions_clearance(self):
        dims = get_hole_dimensions("M4", "through", "clearance")
        assert dims["diameter"] == 4.5  # ISO 273 medium

    def test_hole_dimensions_counterbore(self):
        dims = get_hole_dimensions("M6", "counterbore", "clearance")
        assert "cbore_dia" in dims
        assert dims["cbore_dia"] == 11.25

    def test_hole_dimensions_press(self):
        dims = get_hole_dimensions("M4", "through", "press")
        assert dims["diameter"] == pytest.approx(3.98, rel=1e-3)
