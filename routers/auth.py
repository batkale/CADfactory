import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

import models
import schemas
import security as auth_utils
from database import get_db

logger = logging.getLogger("cadfactory.auth")

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=schemas.Token, status_code=201)
def register(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    logger.info(f"REGISTER attempt — email={user_data.email!r} username={user_data.username!r}")
    try:
        if db.query(models.User).filter(models.User.email == user_data.email).first():
            logger.warning(f"REGISTER rejected — email already registered: {user_data.email!r}")
            raise HTTPException(status_code=409, detail="Email already registered")
        if db.query(models.User).filter(models.User.username == user_data.username).first():
            logger.warning(f"REGISTER rejected — username taken: {user_data.username!r}")
            raise HTTPException(status_code=409, detail="Username already taken")

        hashed = auth_utils.hash_password(user_data.password)
        user = models.User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"REGISTER success — user_id={user.id} email={user.email!r}")

        token = auth_utils.create_access_token({"sub": str(user.id)})
        return {"access_token": token, "token_type": "bearer", "user": user}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"REGISTER error — {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")


@router.post("/login", response_model=schemas.Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    identifier = form_data.username
    logger.info(f"LOGIN attempt — identifier={identifier!r}")
    try:
        user = (
            db.query(models.User)
            .filter(
                (models.User.email == identifier) |
                (models.User.username == identifier)
            )
            .first()
        )
        if not user:
            logger.warning(f"LOGIN failed — no user found for {identifier!r}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        if not auth_utils.verify_password(form_data.password, user.hashed_password):
            logger.warning(f"LOGIN failed — wrong password for {identifier!r}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        if not user.is_active:
            logger.warning(f"LOGIN failed — account disabled for {identifier!r}")
            raise HTTPException(status_code=403, detail="Account disabled")

        token = auth_utils.create_access_token({"sub": str(user.id)})
        logger.info(f"LOGIN success — user_id={user.id} email={user.email!r}")
        return {"access_token": token, "token_type": "bearer", "user": user}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LOGIN error — {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Login failed: {e}")


@router.get("/me", response_model=schemas.UserOut)
def get_me(current_user: models.User = Depends(auth_utils.get_current_user)):
    logger.debug(f"GET /me — user_id={current_user.id}")
    return current_user


@router.delete("/me", status_code=204)
def delete_account(
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
):
    logger.info(f"DELETE account — user_id={current_user.id} email={current_user.email!r}")
    current_user.deleted_at = datetime.now(timezone.utc)
    current_user.is_active = False
    db.commit()
    logger.info(f"DELETE account success (soft) — user_id={current_user.id}")
