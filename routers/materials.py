import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, Depends

import models
import security as auth_utils
from services.cogs import MATERIALS, PROCESSES, REGIONS, VOLUME_TIERS

router = APIRouter(prefix="/materials", tags=["Reference Data"])


@router.get("/")
def list_materials(_: models.User = Depends(auth_utils.get_current_user)):
    """All supported materials with properties."""
    return {
        k: {
            "id": k,
            "name": v["name"],
            "density_g_cm3": v["density_g_cm3"],
            "price_per_kg_usd": v["price_per_kg_usd"],
            "tensile_mpa": v["tensile_mpa"],
            "yield_mpa": v["yield_mpa"],
        }
        for k, v in MATERIALS.items()
    }


@router.get("/processes")
def list_processes(_: models.User = Depends(auth_utils.get_current_user)):
    """All supported manufacturing processes."""
    return {
        k: {
            "id": k,
            "name": v["name"],
            "mrr_cm3_hr": v["mrr_cm3_hr"],
            "setup_hours": v["setup_hours"],
        }
        for k, v in PROCESSES.items()
    }


@router.get("/regions")
def list_regions(_: models.User = Depends(auth_utils.get_current_user)):
    """Manufacturing regions with rates."""
    return {
        k: {
            "id": k,
            "name": v["name"],
            "flag": v["flag"],
            "hourly_rate_usd": v["hourly_rate_usd"],
            "risk": v["risk"],
            "lead_proto": v["lead_days_proto"],
            "lead_volume": v["lead_days_volume"],
            "note": v["note"],
        }
        for k, v in REGIONS.items()
    }


@router.get("/tiers")
def list_tiers(_: models.User = Depends(auth_utils.get_current_user)):
    """Available volume tiers."""
    return {"tiers": VOLUME_TIERS}
