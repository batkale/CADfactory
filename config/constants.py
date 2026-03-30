"""
Centralized constants for CADfactory.

Consolidates all geometric tolerances, manufacturing thresholds,
and physical constants used across geometry parsing, topology
optimization, and engineering math modules.

Inspired by CadQuery's scattered tolerance problem — this module
ensures consistency and discoverability.
"""

from __future__ import annotations

import math

# ── Geometric Tolerances ─────────────────────────────────────────────────────

# General-purpose tolerance for floating-point geometry comparisons (mm)
GEOM_TOL = 1e-6

# Bounding-box containment tolerance (mm) — looser for spatial queries
BBOX_TOL = 1e-2

# STL mesh tolerance for zero-area triangle detection (mm^2)
MESH_ZERO_AREA_TOL = 1e-10

# Ray-casting intersection tolerance (dimensionless parameter space)
RAY_INTERSECT_TOL = 1e-8

# Minimum voxel extent to avoid degenerate grids (mm)
MIN_VOXEL_EXTENT = 1e-6

# Topology optimization convergence threshold (max density change)
SIMP_CONVERGENCE_TOL = 0.01

# OC bisection convergence tolerance
OC_BISECTION_TOL = 1e-4

# Minimum element density (avoids singular stiffness matrix)
SIMP_EMIN = 1e-9

# Maximum element density
SIMP_EMAX = 1.0


# ── Physical Constants ───────────────────────────────────────────────────────

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2          # Golden ratio
INV_PHI = 1 / PHI
R_GAS = 8.314                          # J/(mol*K) universal gas constant
GRAVITY_MM_S2 = 9806.65               # mm/s^2
GRAVITY_M_S2 = 9.80665                # m/s^2


# ── Manufacturing Tolerances ─────────────────────────────────────────────────

# ISO 286 general tolerance grades — IT values in mm for nominal 6-30mm range
ISO_TOLERANCE_GRADES = {
    "IT6": 0.013,
    "IT7": 0.021,
    "IT8": 0.033,
    "IT9": 0.052,
    "IT10": 0.084,
    "IT11": 0.130,
    "IT12": 0.210,
}

# Standard draft angle for injection molding (degrees)
DEFAULT_DRAFT_ANGLE_DEG = 1.5

# Minimum wall thickness by process (mm)
MIN_WALL_THICKNESS = {
    "FDM":          1.5,
    "SLA":          0.5,
    "SLS":          0.8,
    "MJF":          0.8,
    "Injection":    1.0,
    "CNC":          0.8,
    "CastUrethane": 1.0,
    "SheetMetal":   0.8,
}

# STEP file fill factors for volume estimation
STEP_FILL_FACTORS = {
    "assembly": 0.28,
    "complex": 0.38,       # face_count > 50
    "moderate": 0.48,      # face_count > 20
    "simple": 0.62,        # face_count <= 20
}

# High-aspect-ratio fill factor multiplier
HIGH_ASPECT_RATIO_THRESHOLD = 5.0
HIGH_ASPECT_RATIO_FACTOR = 0.75


# ── Complexity Scoring ───────────────────────────────────────────────────────

# STL complexity: surface_area / spherical_equivalent_surface_area
STL_COMPLEXITY_SPHERE_COEFF = 4.836    # ~= (36*pi)^(1/3) for sphere SA from V

# STEP complexity: face_count / divisor, clamped [1, 10]
STEP_COMPLEXITY_DIVISOR = 8.0
COMPLEXITY_MIN = 1.0
COMPLEXITY_MAX = 10.0


# ── Material Database ────────────────────────────────────────────────────────

# Elastic moduli in MPa
ELASTIC_MODULI: dict[str, float] = {
    "ABS":        2200.0,
    "PLA":        3500.0,
    "PETG":       2100.0,
    "Nylon":      2800.0,
    "TPU":         400.0,
    "Al6061":    68900.0,
    "Steel304": 193000.0,
    "Polycarbonate": 2600.0,
    "Resin":      3000.0,
    "HDPE":       1100.0,
}

# Topology optimization material properties {key: (density_gcc, cost_per_kg, display_name)}
TOPO_MATERIALS: dict[str, dict] = {
    "pla":      {"density": 1.24, "cost_per_kg": 25.0,  "name": "PLA"},
    "abs":      {"density": 1.05, "cost_per_kg": 22.0,  "name": "ABS"},
    "petg":     {"density": 1.27, "cost_per_kg": 28.0,  "name": "PETG"},
    "nylon":    {"density": 1.14, "cost_per_kg": 45.0,  "name": "Nylon"},
    "al6061":   {"density": 2.70, "cost_per_kg": 5.50,  "name": "Al 6061-T6"},
    "steel":    {"density": 7.85, "cost_per_kg": 1.20,  "name": "Steel"},
    "titanium": {"density": 4.43, "cost_per_kg": 80.0,  "name": "Titanium"},
}


# ── Execution Limits ─────────────────────────────────────────────────────────

# CadQuery script execution
MAX_SCRIPT_LENGTH = 10_000             # characters
DEFAULT_CADQUERY_TIMEOUT = 60          # seconds

# Topology optimization
DEFAULT_SIMP_RESOLUTION = 20
MAX_SIMP_RESOLUTION = 40
MIN_SIMP_RESOLUTION = 8
DEFAULT_SIMP_VOLFRAC = 0.4
DEFAULT_SIMP_PENAL = 3.0
DEFAULT_SIMP_RMIN = 1.5
DEFAULT_SIMP_MAX_ITER = 80

# FDM print speed estimate (g/min)
FDM_PRINT_SPEED_G_PER_MIN = 10.0 / 60.0

# File cleanup
DEFAULT_FILE_MAX_AGE_HOURS = 24
CLEANUP_INTERVAL_HOURS = 6
