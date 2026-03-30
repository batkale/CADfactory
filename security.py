import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

import models
from database import get_db

load_dotenv()

logger = logging.getLogger("cadfactory.security")

SECRET_KEY = os.getenv("SECRET_KEY", "changeme-not-for-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 10080))

if SECRET_KEY == "changeme-not-for-production":
    raise RuntimeError(
        "SECRET_KEY is set to the default insecure value. "
        "Set a strong SECRET_KEY in your .env file before starting the server."
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    hashed = pwd_context.hash(password)
    logger.debug("Password hashed successfully")
    return hashed


def verify_password(plain: str, hashed: str) -> bool:
    result = pwd_context.verify(plain, hashed)
    logger.debug(f"Password verify: {'match' if result else 'no match'}")
    return result


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug(f"Token created — sub={data.get('sub')} expires={expire.isoformat()}")
    return token


def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"Token decode failed: {e}")
        return None


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> models.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_token(token)
    if payload is None:
        logger.warning("get_current_user: token decode returned None")
        raise credentials_exception

    user_id = payload.get("sub")
    if user_id is None:
        logger.warning("get_current_user: no 'sub' in token payload")
        raise credentials_exception

    try:
        user = db.query(models.User).filter(models.User.id == int(user_id)).first()
    except Exception as e:
        logger.error(f"get_current_user: DB query failed — {e}", exc_info=True)
        raise credentials_exception

    if user is None:
        logger.warning(f"get_current_user: user_id={user_id} not found in DB")
        raise credentials_exception
    if not user.is_active:
        logger.warning(f"get_current_user: user_id={user_id} is inactive")
        raise credentials_exception

    logger.debug(f"get_current_user: authenticated user_id={user_id}")
    return user
