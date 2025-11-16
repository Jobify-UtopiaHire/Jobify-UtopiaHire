import re
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..config.settings import settings
from ..database.connection import get_db
from ..models.user import User

oauth2_scheme = HTTPBearer(auto_error=False)

PASSWORD_REGEX = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$")

def is_strong_password(pw: str) -> bool:
    return bool(PASSWORD_REGEX.match(pw))

def hash_password(pw: str) -> str:
    # Use bcrypt directly to avoid passlib compatibility issues
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pw.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(pw: str, hashed: str) -> bool:
    # Use bcrypt directly to avoid passlib compatibility issues
    return bcrypt.checkpw(pw.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(tz=timezone.utc) + timedelta(hours=settings.access_expires_hours)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm="HS256")

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(tz=timezone.utc) + timedelta(days=settings.refresh_expires_days)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.refresh_secret, algorithm="HS256")

async def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    if creds is None:
        raise HTTPException(status_code=401, detail="No token")
    token = creds.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = payload.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
