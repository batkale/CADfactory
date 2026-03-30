import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
import security as auth_utils
from database import get_db
from services import cogs as cogs_service
from services import gemini as gemini_service
from services import geometry as geo_service
from services import storage
from services.cache import cache_get, cache_set, make_hash_key

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/run", response_model=schemas.ReportOut, status_code=201)
async def run_analysis(
    req: schemas.AnalysisRequest,
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
):
    # ── 1. Fetch the uploaded file record ─────────────────────
    db_file = db.query(models.UploadedFile).filter(
        models.UploadedFile.id == req.file_id,
        models.UploadedFile.user_id == current_user.id,
    ).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")

    # ── 2. Read raw bytes from disk ───────────────────────────
    try:
        data = await storage.read_file(db_file.stored_filename)
    except FileNotFoundError:
        raise HTTPException(status_code=410, detail="File data no longer available on disk")

    # ── 3. Parse geometry ─────────────────────────────────────
    try:
        geom = geo_service.parse_cad_file(data, db_file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Geometry parse error: {e}")

    # ── 3b. Optional STEP merge ───────────────────────────────
    if req.step_file_id:
        step_file = db.query(models.UploadedFile).filter(
            models.UploadedFile.id == req.step_file_id,
            models.UploadedFile.user_id == current_user.id,
        ).first()
        if step_file:
            try:
                step_data = await storage.read_file(step_file.stored_filename)
                step_geom = geo_service.parse_cad_file(step_data, step_file.filename)
                geom = geo_service.merge_stl_step(geom, step_geom)
            except Exception as e:
                print(f"[STEP merge] Failed: {e} — using STL only")

    # ── 4. Compute COGS ───────────────────────────────────────
    cogs_data = cogs_service.compute_cogs(
        geom,
        material_key=req.material,
        process_key=req.process,
    )

    # ── 5. AI analysis (Gemini, server-side key) ──────────────
    # Cache AI analysis by file hash + material + process to avoid duplicate API calls.
    _ai_cache_key = make_hash_key("gemini_analysis", data) + f":{req.material}:{req.process}"
    cached_ai = cache_get(_ai_cache_key)
    if cached_ai is not None:
        ai_result, ai_used = cached_ai, True
    else:
        ai_result, ai_used = await gemini_service.analyze_geometry(geom, cogs_data)
        if ai_used:
            cache_set(_ai_cache_key, ai_result, ttl=3600)  # cache for 1 hour

    # ── 5b. Real-time price search for BOM parts ───────────────
    bom_list = None
    if geom.bom_items:
        raw_bom = [{
            'idx': b.idx, 'item': b.item, 'description': b.description,
            'qty': b.qty, 'unit_cost': b.unit_cost, 'total_cost': b.total_cost,
            'material': b.material, 'part_type': b.part_type, 'source': b.source,
            'aliexpress_price': b.aliexpress_price, 'amazon_price': b.amazon_price,
            'aliexpress_url': b.aliexpress_url, 'amazon_url': b.amazon_url,
        } for b in geom.bom_items]

        # Search real-time prices for standard parts
        try:
            prices = await gemini_service.search_part_prices(raw_bom)
            for item in raw_bom:
                if item['item'] in prices:
                    p = prices[item['item']]
                    item['aliexpress_price'] = p.get('aliexpress_price')
                    item['amazon_price'] = p.get('amazon_price')
                    item['aliexpress_url'] = p.get('aliexpress_url')
                    item['amazon_url'] = p.get('amazon_url')
                    # Use lowest found price as best estimate
                    found_prices = [x for x in [item['aliexpress_price'], item['amazon_price']] if x]
                    if found_prices:
                        item['unit_cost'] = round(min(found_prices), 4)
                        item['total_cost'] = round(item['unit_cost'] * item['qty'], 4)
        except Exception as e:
            print(f"[Price search] Skipped: {e}")

        bom_list = raw_bom

    # ── 6. Persist report to DB ───────────────────────────────
    report = models.Report(
        user_id=current_user.id,
        file_id=db_file.id,
        # Geometry
        file_format=geom.file_format,
        format_detail=geom.format_detail,
        volume_cm3=geom.volume_cm3,
        surface_area_cm2=geom.surface_area_cm2,
        volume_method=geom.volume_method,
        confidence_interval=geom.confidence_interval,
        triangle_count=geom.triangle_count,
        face_count=geom.face_count,
        bbox_x_mm=geom.bounding_box_mm.x,
        bbox_y_mm=geom.bounding_box_mm.y,
        bbox_z_mm=geom.bounding_box_mm.z,
        complexity_score=geom.complexity_score,
        is_assembly=geom.is_assembly,
        components=geom.components,
        # COGS
        cogs_data=cogs_data,
        # BOM
        bom_items=bom_list,
        # AI
        physics_score=ai_result["physics"]["score"],
        physics_warning=ai_result["physics"]["warning"],
        retail_price_usd=ai_result["economics"]["retail"],
        capital_prototype=ai_result["capital"]["prototype"],
        capital_capex=ai_result["capital"]["capex"],
        capital_total_ask=ai_result["capital"]["total_ask"],
        supply_lead_time=ai_result["supply_chain"]["lead_time"],
        supply_route=ai_result["supply_chain"]["route"],
        supply_risk=ai_result["supply_chain"]["risk"],
        suppliers=ai_result["supply_chain"]["suppliers"],
        optimizations=ai_result["optimizations"],
        ai_used=ai_used,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@router.get("/reports", response_model=List[schemas.ReportSummary])
def list_reports(
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
):
    return (
        db.query(models.Report)
        .filter(
            models.Report.user_id == current_user.id,
            models.Report.deleted_at.is_(None),
        )
        .order_by(models.Report.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/reports/{report_id}", response_model=schemas.ReportOut)
def get_report(
    report_id: int,
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
):
    report = db.query(models.Report).filter(
        models.Report.id == report_id,
        models.Report.user_id == current_user.id,
        models.Report.deleted_at.is_(None),
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.delete("/reports/{report_id}", status_code=204)
def delete_report(
    report_id: int,
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
):
    report = db.query(models.Report).filter(
        models.Report.id == report_id,
        models.Report.user_id == current_user.id,
        models.Report.deleted_at.is_(None),
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    report.deleted_at = datetime.now(timezone.utc)
    db.commit()




@router.post("/quick", status_code=200)
async def quick_analysis(
    file_id: int,
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
):
    """
    Geometry + COGS only (no AI, no DB write).
    Fast preview before committing a full analysis.
    """
    db_file = db.query(models.UploadedFile).filter(
        models.UploadedFile.id == file_id,
        models.UploadedFile.user_id == current_user.id,
    ).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")

    data = await storage.read_file(db_file.stored_filename)
    geom = geo_service.parse_cad_file(data, db_file.filename)
    cogs_data = cogs_service.compute_cogs(geom)

    return {
        "geometry": {
            "format": geom.file_format,
            "format_detail": geom.format_detail,
            "volume_cm3": geom.volume_cm3,
            "surface_area_cm2": geom.surface_area_cm2,
            "volume_method": geom.volume_method,
            "confidence_interval": geom.confidence_interval,
            "bounding_box_mm": {
                "x": geom.bounding_box_mm.x,
                "y": geom.bounding_box_mm.y,
                "z": geom.bounding_box_mm.z,
            },
            "complexity_score": geom.complexity_score,
            "triangle_count": geom.triangle_count,
            "face_count": geom.face_count,
            "is_assembly": geom.is_assembly,
            "components": geom.components,
        },
        "cogs": cogs_data,
    }
