from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database.connection import get_db
from ..models.user import User
from ..schemas.auth_schemas import RegisterRequest, LoginRequest, TokenResponse, UserOut, MeResponse, Message, RefreshRequest, VerifyEmailRequest
from ..utils.security_utils import is_strong_password, hash_password, verify_password, create_access_token, create_refresh_token, get_current_user
from ..middleware.security import (
    limiter,
    sanitize_html,
    sanitize_sql_input,
    validate_email_format,
    is_account_locked,
    record_failed_login,
    clear_failed_logins,
)
import secrets

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/register", response_model=Message, responses={400: {"model": Message}})
@limiter.limit("5/minute")
async def register(request: Request, payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Validate and sanitize inputs
    if not validate_email_format(payload.email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    # Sanitize name and email
    payload.name = sanitize_html(payload.name)
    payload.email = payload.email.lower().strip()
    
    # Validate password
    if not is_strong_password(payload.password):
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters and include lowercase (a-z), uppercase (A-Z), a number (0-9), and a symbol.")
    
    # Check if email already exists
    res = await db.execute(select(User).where(User.email == payload.email))
    existing_user = res.scalar_one_or_none()
    
    if existing_user is not None:
        if not existing_user.email_verified:
            # User exists but email not verified - resend verification email
            verification_token = secrets.token_urlsafe(32)
            existing_user.verification_token = verification_token
            existing_user.name = payload.name  # Update name
            existing_user.password_hash = hash_password(payload.password)  # Update password
            await db.commit()
            
            # Send verification email
            from agent.mailer import send_verification_email
            from ..config.settings import settings
            
            try:
                await send_verification_email(
                    recipient_email=existing_user.email,
                    recipient_name=existing_user.name,
                    verification_token=verification_token,
                    frontend_url=settings.frontend_url
                )
                return Message(message="This email is registered but not verified. We've sent a new verification email to your inbox. Please check your email to verify your account.")
            except Exception as e:
                print(f"⚠️ Failed to send verification email: {str(e)}")
                return Message(message="This email is registered but not verified. Please contact support.")
        else:
            # User exists and is verified
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Generate verification token
    verification_token = secrets.token_urlsafe(32)
    
    # Create user (not verified yet)
    user = User(
        name=payload.name, 
        email=payload.email, 
        password_hash=hash_password(payload.password),
        email_verified=False,
        verification_token=verification_token
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    # Send verification email
    from agent.mailer import send_verification_email
    from ..config.settings import settings
    
    try:
        await send_verification_email(
            recipient_email=user.email,
            recipient_name=user.name,
            verification_token=verification_token,
            frontend_url=settings.frontend_url
        )
    except Exception as e:
        print(f"⚠️ Failed to send verification email: {str(e)}")
        # Don't fail registration if email fails - user can resend later
    
    return Message(message="Registration successful! Please check your email to verify your account.")

@router.post("/verify-email", response_model=TokenResponse, responses={400: {"model": Message}})
async def verify_email(payload: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    # Find user by verification token
    res = await db.execute(select(User).where(User.verification_token == payload.token))
    user = res.scalar_one_or_none()
    
    if user is None:
        # Token not found - it might have been cleared after previous verification
        # This is okay, just show generic error
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")
    
    # If already verified, still log them in (idempotent operation)
    # Don't clear the token yet - keep it for 24 hours in case user clicks link again
    if user.email_verified:
        # Generate tokens for automatic login
        access = create_access_token({"id": user.id, "email": user.email})
        refresh = create_refresh_token({"id": user.id, "email": user.email, "tv": user.token_version})
        return TokenResponse(token=access, refreshToken=refresh, user=UserOut(id=user.id, name=user.name, email=user.email))
    
    # Mark email as verified but DON'T clear token yet (keep it for re-clicks)
    user.email_verified = True
    # user.verification_token = None  # Don't clear - allows link to work multiple times
    await db.commit()
    await db.refresh(user)
    
    # Generate tokens for automatic login
    access = create_access_token({"id": user.id, "email": user.email})
    refresh = create_refresh_token({"id": user.id, "email": user.email, "tv": user.token_version})
    
    return TokenResponse(token=access, refreshToken=refresh, user=UserOut(id=user.id, name=user.name, email=user.email))

@router.post("/resend-verification", response_model=Message, responses={400: {"model": Message}})
@limiter.limit("3/minute")
async def resend_verification(request: Request, payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    # Find user by email
    res = await db.execute(select(User).where(User.email == payload.email))
    user = res.scalar_one_or_none()
    
    if user is None:
        raise HTTPException(status_code=400, detail="User not found")
    
    if user.email_verified:
        raise HTTPException(status_code=400, detail="Email already verified")
    
    # Verify password
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    # Generate new verification token
    verification_token = secrets.token_urlsafe(32)
    user.verification_token = verification_token
    await db.commit()
    
    # Send verification email
    from agent.mailer import send_verification_email
    from ..config.settings import settings
    
    try:
        await send_verification_email(
            recipient_email=user.email,
            recipient_name=user.name,
            verification_token=verification_token,
            frontend_url=settings.frontend_url
        )
    except Exception as e:
        print(f"⚠️ Failed to send verification email: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send verification email")
    
    return Message(message="Verification email sent! Please check your inbox.")

@router.post("/login", response_model=TokenResponse, responses={400: {"model": Message}})
@limiter.limit("10/minute")
async def login(request: Request, payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    # Validate email format
    if not validate_email_format(payload.email):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    # Normalize email
    email = payload.email.lower().strip()
    
    # Check if account is locked
    if is_account_locked(email):
        raise HTTPException(
            status_code=429, 
            detail="Account temporarily locked due to too many failed login attempts. Please try again in 15 minutes."
        )
    
    res = await db.execute(select(User).where(User.email == email))
    user = res.scalar_one_or_none()
    if user is None:
        record_failed_login(email)
        raise HTTPException(status_code=400, detail="Invalid credentials")
    if not verify_password(payload.password, user.password_hash):
        record_failed_login(email)
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    # Check if email is verified
    if not user.email_verified:
        # Generate new verification token and send email
        verification_token = secrets.token_urlsafe(32)
        user.verification_token = verification_token
        await db.commit()
        
        # Send verification email
        from agent.mailer import send_verification_email
        from ..config.settings import settings
        
        try:
            await send_verification_email(
                recipient_email=user.email,
                recipient_name=user.name,
                verification_token=verification_token,
                frontend_url=settings.frontend_url
            )
            raise HTTPException(status_code=403, detail="Please verify your email before logging in. We've sent a new verification link to your inbox.")
        except Exception as e:
            print(f"⚠️ Failed to send verification email: {str(e)}")
            raise HTTPException(status_code=403, detail="Please verify your email before logging in. Check your inbox for the verification link.")
    
    # Clear failed login attempts on successful login
    clear_failed_logins(email)
    
    access = create_access_token({"id": user.id, "email": user.email})
    refresh = create_refresh_token({"id": user.id, "email": user.email, "tv": user.token_version})
    return TokenResponse(token=access, refreshToken=refresh, user=UserOut(id=user.id, name=user.name, email=user.email))

@router.get("/me", response_model=MeResponse, responses={401: {"model": Message}})
async def me(user: User = Depends(get_current_user)):
    return MeResponse(user=UserOut(id=user.id, name=user.name, email=user.email))

@router.post("/refresh", response_model=TokenResponse, responses={401: {"model": Message}})
async def refresh(payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    from jose import jwt, JWTError
    from ..config.settings import settings
    try:
        data = jwt.decode(payload.refreshToken, settings.refresh_secret, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    uid = data.get("id"); tv = data.get("tv")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    res = await db.execute(select(User).where(User.id == uid))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    if tv is None or tv != user.token_version:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    access = create_access_token({"id": user.id, "email": user.email})
    # rotate refresh token by reissuing
    refresh = create_refresh_token({"id": user.id, "email": user.email, "tv": user.token_version})
    return TokenResponse(token=access, refreshToken=refresh, user=UserOut(id=user.id, name=user.name, email=user.email))
