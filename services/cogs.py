import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Parametric COGS Engine — Al 6061-T6, 3-axis CNC.

Model breakdown:
  Material cost  = stock_volume × density × price_per_kg
  Machining time = (stock_vol - part_vol) / MRR × complexity_factor
  Labor          = machining_cost × labor_ratio
  Setup          = fixed_setup / quantity
  Logistics      = weight-based rate per region
  Overhead       = 12% of (machining + labor)

Sources:
  LME Aluminium spot ~$5.50/kg (Q1 2025)
  MRR: 3-axis CNC Al = 1200 cm³/hr (conservative)
  Density Al 6061: 2.70 g/cm³
"""

from services.geometry import GeometryResult  # noqa: E402

# ── Material database ─────────────────────────────────────────

MATERIALS = {
    "Al6061-T6": {
        "name": "Aluminium 6061-T6",
        "density_g_cm3": 2.70,
        "price_per_kg_usd": 5.50,
        "machinability_factor": 1.0,   # reference
        "tensile_mpa": 310,
        "yield_mpa": 276,
    },
    "Al7075-T6": {
        "name": "Aluminium 7075-T6",
        "density_g_cm3": 2.81,
        "price_per_kg_usd": 8.20,
        "machinability_factor": 0.90,
        "tensile_mpa": 572,
        "yield_mpa": 503,
    },
    "SS304": {
        "name": "Stainless Steel 304",
        "density_g_cm3": 8.00,
        "price_per_kg_usd": 4.80,
        "machinability_factor": 0.45,
        "tensile_mpa": 515,
        "yield_mpa": 205,
    },
    "Titanium_Grade5": {
        "name": "Titanium Grade 5 (Ti-6Al-4V)",
        "density_g_cm3": 4.43,
        "price_per_kg_usd": 35.0,
        "machinability_factor": 0.30,
        "tensile_mpa": 950,
        "yield_mpa": 880,
    },
    "PEEK": {
        "name": "PEEK (polymer)",
        "density_g_cm3": 1.32,
        "price_per_kg_usd": 95.0,
        "machinability_factor": 1.40,
        "tensile_mpa": 100,
        "yield_mpa": 91,
    },
}

# ── Process database ──────────────────────────────────────────
#
# "type" controls the cost model used in compute_cogs:
#   "subtractive"  — stock purchased, material removed (CNC / turning)
#   "additive"     — only the part volume of material consumed (FDM / SLA / SLS)
#   "molding"      — injection: high fixed tooling, very low marginal material cost

PROCESSES = {
    "CNC_3axis": {
        "name": "3-Axis CNC Milling",
        "type": "subtractive",
        "mrr_cm3_hr": 1200,        # material removal rate
        "setup_hours": 0.75,       # per job setup (not per unit)
        "overhead_ratio": 0.12,
    },
    "CNC_5axis": {
        "name": "5-Axis CNC Milling",
        "type": "subtractive",
        "mrr_cm3_hr": 900,
        "setup_hours": 1.5,
        "overhead_ratio": 0.15,
    },
    "Turning": {
        "name": "CNC Turning",
        "type": "subtractive",
        "mrr_cm3_hr": 2000,
        "setup_hours": 0.5,
        "overhead_ratio": 0.10,
    },
    # ── Additive ─────────────────────────────────────────────
    "FDM": {
        "name": "FDM 3D Printing",
        "type": "additive",
        # Build speed: ~20 cm³/hr at 0.2 mm layer height, 40 mm/s
        "build_rate_cm3_hr": 20,
        "support_factor": 1.15,    # ~15% extra material for supports
        "setup_hours": 0.25,
        "overhead_ratio": 0.10,
        # Compatible materials: PLA, ABS, PETG, Nylon, TPU
        "compatible_materials": {"PLA", "ABS", "PETG", "Nylon", "TPU"},
    },
    "SLA": {
        "name": "SLA Resin Printing",
        "type": "additive",
        "build_rate_cm3_hr": 10,
        "support_factor": 1.10,
        "setup_hours": 0.50,       # post-cure wash adds time
        "overhead_ratio": 0.14,
        "compatible_materials": {"Standard_Resin", "Engineering_Resin", "Castable_Resin"},
    },
    "SLS": {
        "name": "SLS Powder Bed Fusion",
        "type": "additive",
        "build_rate_cm3_hr": 30,
        "support_factor": 1.0,     # self-supporting in powder bed
        "setup_hours": 1.0,        # powder loading + depowdering
        "overhead_ratio": 0.16,
        "compatible_materials": {"Nylon_PA12", "Nylon_PA11", "TPU_SLS"},
    },
    # ── Injection Molding ─────────────────────────────────────
    "InjectionMolding": {
        "name": "Injection Molding",
        "type": "molding",
        # Cycle time: ~30 sec/shot → 120 shots/hr
        "shots_per_hr": 120,
        # Tooling (mold) cost in USD — amortised over production run
        "tooling_cost_usd": 15_000,
        "setup_hours": 4.0,
        "overhead_ratio": 0.08,
        "compatible_materials": {"PP", "ABS", "PE", "Nylon_PA66", "PC"},
    },
}

# ── Additive-process material overrides ───────────────────────
# These materials are relevant for additive / molding but not CNC.
# Density and price extend the MATERIALS dict for cost computation.
ADDITIVE_MATERIALS = {
    "PLA": {
        "name": "PLA (FDM)",
        "density_g_cm3": 1.24,
        "price_per_kg_usd": 25.0,
        "machinability_factor": 1.0,
        "tensile_mpa": 50,
        "yield_mpa": 45,
    },
    "ABS": {
        "name": "ABS (FDM / Injection)",
        "density_g_cm3": 1.05,
        "price_per_kg_usd": 22.0,
        "machinability_factor": 1.0,
        "tensile_mpa": 40,
        "yield_mpa": 35,
    },
    "PETG": {
        "name": "PETG (FDM)",
        "density_g_cm3": 1.27,
        "price_per_kg_usd": 28.0,
        "machinability_factor": 1.0,
        "tensile_mpa": 53,
        "yield_mpa": 50,
    },
    "Nylon": {
        "name": "Nylon PA12 (FDM)",
        "density_g_cm3": 1.01,
        "price_per_kg_usd": 60.0,
        "machinability_factor": 1.0,
        "tensile_mpa": 48,
        "yield_mpa": 42,
    },
    "Nylon_PA12": {
        "name": "Nylon PA12 (SLS)",
        "density_g_cm3": 1.01,
        "price_per_kg_usd": 80.0,
        "machinability_factor": 1.0,
        "tensile_mpa": 48,
        "yield_mpa": 42,
    },
    "Nylon_PA11": {
        "name": "Nylon PA11 (SLS)",
        "density_g_cm3": 1.03,
        "price_per_kg_usd": 90.0,
        "machinability_factor": 1.0,
        "tensile_mpa": 52,
        "yield_mpa": 46,
    },
    "Standard_Resin": {
        "name": "Standard Resin (SLA)",
        "density_g_cm3": 1.18,
        "price_per_kg_usd": 50.0,
        "machinability_factor": 1.0,
        "tensile_mpa": 65,
        "yield_mpa": 58,
    },
    "Engineering_Resin": {
        "name": "Engineering Resin (SLA)",
        "density_g_cm3": 1.20,
        "price_per_kg_usd": 150.0,
        "machinability_factor": 1.0,
        "tensile_mpa": 90,
        "yield_mpa": 80,
    },
    "PP": {
        "name": "Polypropylene (Injection)",
        "density_g_cm3": 0.91,
        "price_per_kg_usd": 1.80,
        "machinability_factor": 1.0,
        "tensile_mpa": 35,
        "yield_mpa": 30,
    },
    "Nylon_PA66": {
        "name": "Nylon PA66 (Injection)",
        "density_g_cm3": 1.14,
        "price_per_kg_usd": 4.50,
        "machinability_factor": 1.0,
        "tensile_mpa": 82,
        "yield_mpa": 70,
    },
    "PC": {
        "name": "Polycarbonate (Injection)",
        "density_g_cm3": 1.20,
        "price_per_kg_usd": 3.50,
        "machinability_factor": 1.0,
        "tensile_mpa": 65,
        "yield_mpa": 60,
    },
}

# ── Region rates ─────────────────────────────────────────────

REGIONS = {
    "local": {
        "name": "Local Hub",
        "flag": "🏭",
        "hourly_rate_usd": 85,
        "setup_cost_usd": 380,
        "logistics_base_usd": 9,
        "logistics_per_kg_usd": 0.80,
        "material_markup": 1.10,
        "lead_days_proto": "5–10",
        "lead_days_volume": "10–20",
        "risk": "Low",
        "note": "Best QC. Fastest iteration. Ideal for regulated sectors.",
    },
    "nearshore": {
        "name": "Nearshore",
        "flag": "🌍",
        "hourly_rate_usd": 45,
        "setup_cost_usd": 220,
        "logistics_base_usd": 20,
        "logistics_per_kg_usd": 1.20,
        "material_markup": 1.00,
        "lead_days_proto": "12–18",
        "lead_days_volume": "20–35",
        "risk": "Low–Med",
        "note": "Best cost-quality balance at 100+ units.",
    },
    "offshore": {
        "name": "Offshore",
        "flag": "🚢",
        "hourly_rate_usd": 22,
        "setup_cost_usd": 160,
        "logistics_base_usd": 38,
        "logistics_per_kg_usd": 2.50,
        "material_markup": 0.88,
        "lead_days_proto": "21–35",
        "lead_days_volume": "40–60",
        "risk": "Medium",
        "note": "Lowest unit cost. Factor 3–5% tariffs.",
    },
}

VOLUME_TIERS = [1, 10, 50, 100, 500, 1000]


def _r2(n: float) -> float:
    return round(n, 2)


def compute_cogs(
    geom: GeometryResult,
    material_key: str = "Al6061-T6",
    process_key: str = "CNC_3axis",
) -> dict:
    # Resolve material — check additive materials too
    mat = MATERIALS.get(material_key) or ADDITIVE_MATERIALS.get(material_key, MATERIALS["Al6061-T6"])
    proc = PROCESSES.get(process_key, PROCESSES["CNC_3axis"])
    proc_type = proc.get("type", "subtractive")
    ci = geom.confidence_interval

    part_weight_kg = (geom.volume_cm3 * mat["density_g_cm3"]) / 1000.0

    if proc_type == "subtractive":
        return _compute_subtractive(geom, mat, proc, ci, part_weight_kg)
    elif proc_type == "additive":
        return _compute_additive(geom, mat, proc, ci, part_weight_kg)
    elif proc_type == "molding":
        return _compute_molding(geom, mat, proc, ci, part_weight_kg)
    else:
        return _compute_subtractive(geom, mat, proc, ci, part_weight_kg)


def _compute_subtractive(geom, mat, proc, ci, part_weight_kg) -> dict:
    """CNC milling / turning: buy stock block, remove material."""
    is_assembly = geom.is_assembly
    stock_multiplier = 1.15 if is_assembly else 1.45
    stock_volume_cm3 = geom.volume_cm3 * stock_multiplier
    removed_volume_cm3 = stock_volume_cm3 - geom.volume_cm3

    weight_kg = (stock_volume_cm3 * mat["density_g_cm3"]) / 1000.0
    base_material_usd = weight_kg * mat["price_per_kg_usd"]

    complexity_factor = min(max(geom.complexity_score * 0.45, 0.5), 3.5)
    mrr_effective = proc["mrr_cm3_hr"] * mat["machinability_factor"]
    machining_hours = max(
        (removed_volume_cm3 / mrr_effective) * complexity_factor, 0.05
    )

    regions_out = {}
    for rk, region in REGIONS.items():
        material_usd = _r2(base_material_usd * region["material_markup"])
        machining_usd = _r2(machining_hours * region["hourly_rate_usd"])
        labor_usd = _r2(machining_usd * 0.18)
        overhead_usd = _r2((machining_usd + labor_usd) * proc["overhead_ratio"])
        logistics_usd = _r2(
            region["logistics_base_usd"] + part_weight_kg * region["logistics_per_kg_usd"]
        )
        tiers = _build_tiers(material_usd, machining_usd, labor_usd, overhead_usd, logistics_usd, region, ci)
        regions_out[rk] = _region_meta(region, tiers)

    return {
        "regions": regions_out,
        "material_grade": mat["name"],
        "process": proc["name"],
        "process_type": "subtractive",
        "confidence_interval": ci,
        "stock_volume_cm3": _r2(stock_volume_cm3),
        "machining_hours_per_unit": _r2(machining_hours),
        "part_weight_kg": _r2(part_weight_kg),
        "notes": (
            f"MRR: {mrr_effective:.0f} cm³/hr · "
            f"Complexity factor: {complexity_factor:.2f} · "
            f"CI: ±{int(ci*100)}%"
        ),
    }


def _compute_additive(geom, mat, proc, ci, part_weight_kg) -> dict:
    """FDM / SLA / SLS: consume only part volume × support factor."""
    consumed_volume_cm3 = geom.volume_cm3 * proc.get("support_factor", 1.0)
    consumed_weight_kg = (consumed_volume_cm3 * mat["density_g_cm3"]) / 1000.0
    base_material_usd = consumed_weight_kg * mat["price_per_kg_usd"]

    # Build time drives machine cost
    complexity_factor = min(max(geom.complexity_score * 0.3, 0.5), 2.0)
    build_hours = max(
        (consumed_volume_cm3 / proc["build_rate_cm3_hr"]) * complexity_factor,
        0.1,
    )
    # Add fixed setup time amortised later
    setup_hours = proc.get("setup_hours", 0.25)

    regions_out = {}
    for rk, region in REGIONS.items():
        material_usd = _r2(base_material_usd * region["material_markup"])
        machine_usd = _r2(build_hours * region["hourly_rate_usd"])
        labor_usd = _r2(machine_usd * 0.12)
        overhead_usd = _r2((machine_usd + labor_usd) * proc["overhead_ratio"])
        logistics_usd = _r2(
            region["logistics_base_usd"] + part_weight_kg * region["logistics_per_kg_usd"]
        )
        tiers = _build_tiers(material_usd, machine_usd, labor_usd, overhead_usd, logistics_usd, region, ci)
        regions_out[rk] = _region_meta(region, tiers)

    return {
        "regions": regions_out,
        "material_grade": mat["name"],
        "process": proc["name"],
        "process_type": "additive",
        "confidence_interval": ci,
        "stock_volume_cm3": _r2(consumed_volume_cm3),
        "machining_hours_per_unit": _r2(build_hours),
        "part_weight_kg": _r2(part_weight_kg),
        "notes": (
            f"Build rate: {proc['build_rate_cm3_hr']} cm³/hr · "
            f"Complexity factor: {complexity_factor:.2f} · "
            f"CI: ±{int(ci*100)}%"
        ),
    }


def _compute_molding(geom, mat, proc, ci, part_weight_kg) -> dict:
    """Injection molding: high tooling cost amortised, very low marginal material."""
    base_material_usd = part_weight_kg * mat["price_per_kg_usd"]
    tooling_cost = proc.get("tooling_cost_usd", 15_000)
    shots_per_hr = proc.get("shots_per_hr", 120)
    cycle_hours = 1.0 / shots_per_hr  # hours per shot

    regions_out = {}
    for rk, region in REGIONS.items():
        material_usd = _r2(base_material_usd * region["material_markup"])
        machine_usd = _r2(cycle_hours * region["hourly_rate_usd"])
        labor_usd = _r2(machine_usd * 0.10)
        overhead_usd = _r2((machine_usd + labor_usd) * proc["overhead_ratio"])
        logistics_usd = _r2(
            region["logistics_base_usd"] + part_weight_kg * region["logistics_per_kg_usd"]
        )

        tiers = {}
        for qty in VOLUME_TIERS:
            vol_discount = 1.0
            if qty >= 1000:
                vol_discount = 0.90
            elif qty >= 500:
                vol_discount = 0.94
            elif qty >= 100:
                vol_discount = 0.97

            mat_discounted = _r2(material_usd * vol_discount)
            # Tooling amortised over entire run
            tooling_per_unit = _r2(tooling_cost / max(qty, 1))
            # Setup per unit
            setup_usd = _r2((region["setup_cost_usd"] + proc.get("setup_hours", 4.0) * region["hourly_rate_usd"]) / qty)

            total = _r2(
                mat_discounted + machine_usd + labor_usd + overhead_usd + logistics_usd + tooling_per_unit + setup_usd
            )
            tiers[str(qty)] = {
                "material": mat_discounted,
                "machining": machine_usd,
                "labor": labor_usd,
                "overhead": overhead_usd,
                "logistics": logistics_usd,
                "tooling": tooling_per_unit,
                "setup": setup_usd,
                "total": total,
                "low": _r2(total * (1 - ci)),
                "high": _r2(total * (1 + ci)),
            }

        regions_out[rk] = _region_meta(region, tiers)

    return {
        "regions": regions_out,
        "material_grade": mat["name"],
        "process": proc["name"],
        "process_type": "molding",
        "confidence_interval": ci,
        "stock_volume_cm3": _r2(geom.volume_cm3),
        "machining_hours_per_unit": _r2(cycle_hours),
        "part_weight_kg": _r2(part_weight_kg),
        "tooling_cost_usd": tooling_cost,
        "notes": (
            f"Tooling: ${tooling_cost:,} · "
            f"Cycle: {cycle_hours*3600:.1f}s · "
            f"CI: ±{int(ci*100)}%"
        ),
    }


def _build_tiers(material_usd, process_usd, labor_usd, overhead_usd, logistics_usd, region, ci) -> dict:
    tiers = {}
    for qty in VOLUME_TIERS:
        vol_discount = 1.0
        if qty >= 100:
            vol_discount = 0.92
        elif qty >= 50:
            vol_discount = 0.96
        elif qty >= 10:
            vol_discount = 0.98

        mat_discounted = _r2(material_usd * vol_discount)
        setup_usd = _r2(region["setup_cost_usd"] / qty)

        total = _r2(
            mat_discounted + process_usd + labor_usd + overhead_usd + logistics_usd + setup_usd
        )
        tiers[str(qty)] = {
            "material": mat_discounted,
            "machining": process_usd,
            "labor": labor_usd,
            "overhead": overhead_usd,
            "logistics": logistics_usd,
            "setup": setup_usd,
            "total": total,
            "low": _r2(total * (1 - ci)),
            "high": _r2(total * (1 + ci)),
        }
    return tiers


def _region_meta(region: dict, tiers: dict) -> dict:
    return {
        "name": region["name"],
        "flag": region["flag"],
        "lead": region["lead_days_proto"],
        "lead_volume": region["lead_days_volume"],
        "risk": region["risk"],
        "note": region["note"],
        "tiers": tiers,
    }
