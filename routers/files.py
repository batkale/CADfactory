import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

import models
import schemas
import security as auth_utils
from database import get_db
from services import storage

router = APIRouter(prefix="/files", tags=["Files"])


@router.post("/upload", response_model=schemas.FileOut, status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
):
    data = await file.read()

    # Validate
    try:
        storage.validate_file(file.filename, len(data))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Save to disk
    stored_name = storage.generate_stored_filename(file.filename)
    upload_path = await storage.save_file(data, stored_name)

    ext = Path(file.filename).suffix.lower().lstrip(".")
    file_format = {"stl": "STL", "3mf": "3MF", "step": "STEP", "stp": "STEP"}.get(ext, ext.upper())

    db_file = models.UploadedFile(
        user_id=current_user.id,
        filename=file.filename,
        stored_filename=stored_name,
        file_format=file_format,
        file_size_bytes=len(data),
        upload_path=upload_path,
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    return db_file


@router.get("/", response_model=List[schemas.FileOut])
def list_files(
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
):
    return (
        db.query(models.UploadedFile)
        .filter(
            models.UploadedFile.user_id == current_user.id,
            models.UploadedFile.deleted_at.is_(None),
        )
        .order_by(models.UploadedFile.uploaded_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/{file_id}", response_model=schemas.FileOut)
def get_file(
    file_id: int,
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
):
    db_file = db.query(models.UploadedFile).filter(
        models.UploadedFile.id == file_id,
        models.UploadedFile.user_id == current_user.id,
        models.UploadedFile.deleted_at.is_(None),
    ).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    return db_file


@router.get("/{file_id}/download")
def download_file(
    file_id: int,
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
):
    db_file = db.query(models.UploadedFile).filter(
        models.UploadedFile.id == file_id,
        models.UploadedFile.user_id == current_user.id,
        models.UploadedFile.deleted_at.is_(None),
    ).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    if not os.path.exists(db_file.upload_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    media_types = {"STL": "application/sla", "3MF": "application/vnd.ms-package.3dmanufacturing-3dmodel+xml", "STEP": "application/step"}
    media_type = media_types.get(db_file.file_format, "application/octet-stream")
    return FileResponse(db_file.upload_path, media_type=media_type, filename=db_file.filename)


@router.delete("/{file_id}", status_code=204)
def delete_file(
    file_id: int,
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
):
    db_file = db.query(models.UploadedFile).filter(
        models.UploadedFile.id == file_id,
        models.UploadedFile.user_id == current_user.id,
        models.UploadedFile.deleted_at.is_(None),
    ).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")

    db_file.deleted_at = datetime.now(timezone.utc)
    db.commit()
