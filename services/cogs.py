import sys, os
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

import math
from dataclasses import dataclass, field
from typing import Dict
from services.geometry import GeometryResult


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

PROCESSES = {
    "CNC_3axis": {
        "name": "3-Axis CNC Milling",
        "mrr_cm3_hr": 1200,        # material removal rate
        "setup_hours": 0.75,       # per job setup (not per unit)
        "overhead_ratio": 0.12,
    },
    "CNC_5axis": {
        "name": "5-Axis CNC Milling",
        "mrr_cm3_hr": 900,
        "setup_hours": 1.5,
        "overhead_ratio": 0.15,
    },
    "Turning": {
        "name": "CNC Turning",
        "mrr_cm3_hr": 2000,
        "setup_hours": 0.5,
        "overhead_ratio": 0.10,
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

    mat = MATERIALS.get(material_key, MATERIALS["Al6061-T6"])
    proc = PROCESSES.get(process_key, PROCESSES["CNC_3axis"])
    ci = geom.confidence_interval

    # ── Stock volume (part + ~45% removal for prismatic parts) ──
    is_assembly = geom.is_assembly
    stock_multiplier = 1.15 if is_assembly else 1.45
    stock_volume_cm3 = geom.volume_cm3 * stock_multiplier
    removed_volume_cm3 = stock_volume_cm3 - geom.volume_cm3

    # ── Material cost per unit ───────────────────────────────
    weight_kg = (stock_volume_cm3 * mat["density_g_cm3"]) / 1000.0
    base_material_usd = weight_kg * mat["price_per_kg_usd"]

    # ── Machining time ────────────────────────────────────────
    # Complexity factor: higher complexity = slower effective MRR
    complexity_factor = min(max(geom.complexity_score * 0.45, 0.5), 3.5)
    # Time = removed volume / (MRR × machinability) × complexity
    mrr_effective = proc["mrr_cm3_hr"] * mat["machinability_factor"]
    machining_hours = max(
        (removed_volume_cm3 / mrr_effective) * complexity_factor, 0.05
    )

    # ── Part weight for logistics ─────────────────────────────
    part_weight_kg = (geom.volume_cm3 * mat["density_g_cm3"]) / 1000.0

    # ── Build region × tier matrix ────────────────────────────
    regions_out = {}

    for rk, region in REGIONS.items():
        material_usd = _r2(base_material_usd * region["material_markup"])
        machining_usd = _r2(machining_hours * region["hourly_rate_usd"])
        labor_usd = _r2(machining_usd * 0.18)
        overhead_usd = _r2((machining_usd + labor_usd) * proc["overhead_ratio"])
        logistics_usd = _r2(
            region["logistics_base_usd"]
            + part_weight_kg * region["logistics_per_kg_usd"]
        )

        tiers = {}
        for qty in VOLUME_TIERS:
            # Volume discount on material (bulk buying)
            vol_discount = 1.0
            if qty >= 100:
                vol_discount = 0.92
            elif qty >= 50:
                vol_discount = 0.96
            elif qty >= 10:
                vol_discount = 0.98

            mat_discounted = _r2(material_usd * vol_discount)

            # Setup amortised over quantity
            setup_usd = _r2(region["setup_cost_usd"] / qty)

            total = _r2(
                mat_discounted
                + machining_usd
                + labor_usd
                + overhead_usd
                + logistics_usd
                + setup_usd
            )

            tiers[str(qty)] = {
                "material": mat_discounted,
                "machining": machining_usd,
                "labor": labor_usd,
                "overhead": overhead_usd,
                "logistics": logistics_usd,
                "setup": setup_usd,
                "total": total,
                "low": _r2(total * (1 - ci)),
                "high": _r2(total * (1 + ci)),
            }

        regions_out[rk] = {
            "name": region["name"],
            "flag": region["flag"],
            "lead": region["lead_days_proto"],
            "lead_volume": region["lead_days_volume"],
            "risk": region["risk"],
            "note": region["note"],
            "tiers": tiers,
        }

    return {
        "regions": regions_out,
        "material_grade": mat["name"],
        "process": proc["name"],
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
