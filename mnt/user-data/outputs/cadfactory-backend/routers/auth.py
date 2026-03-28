import auth as auth_utils
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=schemas.Token, status_code=201)
def register(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check duplicates
    if db.query(models.User).filter(models.User.email == user_data.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    if db.query(models.User).filter(models.User.username == user_data.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")

    user = models.User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=auth_utils.hash_password(user_data.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = auth_utils.create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer", "user": user}


@router.post("/login", response_model=schemas.Token)
def login(credentials: schemas.UserLogin, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == credentials.email).first()
    if not user or not auth_utils.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    token = auth_utils.create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer", "user": user}


@router.get("/me", response_model=schemas.UserOut)
def get_me(current_user: models.User = Depends(auth_utils.get_current_user)):
    return current_user


@router.delete("/me", status_code=204)
def delete_account(
    current_user: models.User = Depends(auth_utils.get_current_user),
    db: Session = Depends(get_db),
):
    db.delete(current_user)
    db.commit()
