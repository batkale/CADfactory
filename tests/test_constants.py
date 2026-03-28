"""
Unit tests for config/constants.py — validate constant values and consistency.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import (
    BBOX_TOL,
    COMPLEXITY_MAX,
    COMPLEXITY_MIN,
    DEFAULT_CADQUERY_TIMEOUT,
    DEFAULT_DRAFT_ANGLE_DEG,
    DEFAULT_SIMP_RESOLUTION,
    ELASTIC_MODULI,
    GEOM_TOL,
    GRAVITY_M_S2,
    GRAVITY_MM_S2,
    INV_PHI,
    ISO_TOLERANCE_GRADES,
    MAX_SCRIPT_LENGTH,
    MAX_SIMP_RESOLUTION,
    MESH_ZERO_AREA_TOL,
    MIN_SIMP_RESOLUTION,
    MIN_VOXEL_EXTENT,
    MIN_WALL_THICKNESS,
    OC_BISECTION_TOL,
    PHI,
    PI,
    R_GAS,
    RAY_INTERSECT_TOL,
    SIMP_CONVERGENCE_TOL,
    SIMP_EMAX,
    SIMP_EMIN,
    STEP_FILL_FACTORS,
    STL_COMPLEXITY_SPHERE_COEFF,
    TOPO_MATERIALS,
)


class TestPhysicalConstants:
    def test_pi(self):
        assert PI == pytest.approx(math.pi, rel=1e-15)

    def test_phi(self):
        assert PHI == pytest.approx((1 + math.sqrt(5)) / 2, rel=1e-15)

    def test_inv_phi(self):
        assert PHI * INV_PHI == pytest.approx(1.0, rel=1e-15)

    def test_r_gas(self):
        assert R_GAS == pytest.approx(8.314, rel=1e-3)

    def test_gravity(self):
        assert GRAVITY_M_S2 == pytest.approx(9.80665, rel=1e-5)
        assert GRAVITY_MM_S2 == pytest.approx(GRAVITY_M_S2 * 1000, rel=1e-5)


class TestToleranceHierarchy:
    """Tolerances should be ordered: tighter tolerances have smaller values."""

    def test_geom_tighter_than_bbox(self):
        assert GEOM_TOL < BBOX_TOL

    def test_mesh_zero_area_very_tight(self):
        assert MESH_ZERO_AREA_TOL < GEOM_TOL

    def test_all_tolerances_positive(self):
        for tol in [GEOM_TOL, BBOX_TOL, MESH_ZERO_AREA_TOL, RAY_INTERSECT_TOL,
                     MIN_VOXEL_EXTENT, SIMP_CONVERGENCE_TOL, OC_BISECTION_TOL]:
            assert tol > 0

    def test_simp_emin_less_than_emax(self):
        assert SIMP_EMIN < SIMP_EMAX


class TestManufacturingConstants:
    def test_draft_angle_positive(self):
        assert DEFAULT_DRAFT_ANGLE_DEG > 0

    def test_wall_thickness_all_positive(self):
        for process, thickness in MIN_WALL_THICKNESS.items():
            assert thickness > 0, f"{process} wall thickness must be positive"

    def test_iso_tolerance_grades_ordered(self):
        grades = sorted(ISO_TOLERANCE_GRADES.items(), key=lambda x: int(x[0][2:]))
        for i in range(len(grades) - 1):
            assert grades[i][1] < grades[i + 1][1], (
                f"IT{grades[i][0]} should be tighter than IT{grades[i+1][0]}"
            )


class TestStepFillFactors:
    def test_assembly_smallest(self):
        assert STEP_FILL_FACTORS["assembly"] < STEP_FILL_FACTORS["simple"]

    def test_all_between_0_and_1(self):
        for key, val in STEP_FILL_FACTORS.items():
            assert 0 < val < 1, f"{key} fill factor out of range"


class TestMaterials:
    def test_elastic_moduli_positive(self):
        for mat, E in ELASTIC_MODULI.items():
            assert E > 0, f"{mat} modulus must be positive"

    def test_topo_materials_have_required_keys(self):
        for key, props in TOPO_MATERIALS.items():
            assert "density" in props, f"{key} missing density"
            assert "cost_per_kg" in props, f"{key} missing cost_per_kg"
            assert "name" in props, f"{key} missing name"
            assert props["density"] > 0
            assert props["cost_per_kg"] > 0


class TestExecutionLimits:
    def test_script_length_reasonable(self):
        assert MAX_SCRIPT_LENGTH >= 1000

    def test_timeout_reasonable(self):
        assert DEFAULT_CADQUERY_TIMEOUT >= 10

    def test_resolution_bounds(self):
        assert MIN_SIMP_RESOLUTION < DEFAULT_SIMP_RESOLUTION < MAX_SIMP_RESOLUTION


class TestComplexityScoring:
    def test_complexity_range(self):
        assert COMPLEXITY_MIN >= 0
        assert COMPLEXITY_MAX > COMPLEXITY_MIN

    def test_sphere_coeff_positive(self):
        assert STL_COMPLEXITY_SPHERE_COEFF > 0
