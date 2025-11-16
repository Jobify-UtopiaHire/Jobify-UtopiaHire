from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from authlib.integrations.starlette_client import OAuth, OAuthError
from ..config.settings import settings
from ..database.connection import get_db
from ..models.user import User
from ..utils.security_utils import create_access_token, create_refresh_token
from ..schemas.auth_schemas import UserOut
import secrets

router = APIRouter(prefix="/api/auth/oauth", tags=["oauth"])

# Initialize OAuth client
oauth = OAuth()

# Configure Google OAuth
oauth.register(
    name='google',
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# Configure GitHub OAuth
oauth.register(
    name='github',
    client_id=settings.github_client_id,
    client_secret=settings.github_client_secret,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    client_kwargs={'scope': 'user:email'}
)


@router.get("/google")
async def google_login(
    request: Request,
    login_hint: str | None = Query(default=None, description="Suggest an email to Google account chooser"),
):
    """Redirect user to Google OAuth consent screen (always shows account chooser).
    Accepts optional login_hint to pre-fill a suggested email.
    """
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    
    redirect_uri = settings.google_redirect_uri
    # Force Google to show the account chooser every time; include optional login_hint
    params = { 'prompt': 'select_account' }
    if login_hint:
        params['login_hint'] = login_hint
    return await oauth.google.authorize_redirect(request, redirect_uri, **params)


@router.get("/google/callback")
async def google_callback(request: Request, db: AsyncSession = Depends(get_db)):
    """Handle Google OAuth callback"""
    try:
        print("📥 Google OAuth callback received")
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if not user_info:
            print("❌ Failed to get user info from Google")
            raise HTTPException(status_code=400, detail="Failed to get user info from Google")
        
        email = user_info.get('email')
        name = user_info.get('name', email.split('@')[0] if email else 'User')
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by Google")
        
        # Find or create user
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if user is None:
            # Create new user with OAuth (no password)
            # OAuth users are automatically verified since the OAuth provider verified the email
            user = User(
                name=name,
                email=email,
                password_hash=secrets.token_urlsafe(32),  # Random hash, won't be used
                email_verified=True,  # OAuth users are pre-verified
                verification_token=None
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        
        # Generate tokens
        access = create_access_token({"id": user.id, "email": user.email})
        refresh = create_refresh_token({"id": user.id, "email": user.email, "tv": user.token_version})
        
        # Redirect back to frontend with tokens in URL fragments (more secure than query params)
        frontend_url = settings.frontend_url
        redirect_url = f"{frontend_url}?token={access}&refreshToken={refresh}"
        print(f"🔑 OAuth Success - Redirecting to: {frontend_url}?token=***&refreshToken=***")
        print(f"👤 User: {user.email} (ID: {user.id})")
        return RedirectResponse(url=redirect_url)
        
    except OAuthError as e:
        print(f"❌ OAuth Error: {str(e)}")
        # Redirect to frontend with error
        frontend_url = settings.frontend_url
        error_url = f"{frontend_url}?error=oauth_failed&message={str(e)}"
        return RedirectResponse(url=error_url)


@router.get("/github")
async def github_login(
    request: Request,
    login: str | None = Query(default=None, description="Suggest a GitHub username to login with"),
    allow_signup: bool = Query(default=True, description="Allow GitHub to show signup option"),
):
    """Redirect user to GitHub OAuth screen.
    Note: GitHub does not support a true account chooser like Google.
    We pass prompt=consent (ignored by GitHub but harmless), and support optional 'login' to hint a username.
    """
    if not settings.github_client_id or not settings.github_client_secret:
        raise HTTPException(status_code=500, detail="GitHub OAuth not configured")
    
    redirect_uri = settings.github_redirect_uri
    params = {
        # GitHub ignores unknown params, but including 'prompt=consent' is harmless
        'prompt': 'consent',
        'allow_signup': 'true' if allow_signup else 'false',
    }
    if login:
        params['login'] = login
    return await oauth.github.authorize_redirect(request, redirect_uri, **params)


@router.get("/github/callback")
async def github_callback(request: Request, db: AsyncSession = Depends(get_db)):
    """Handle GitHub OAuth callback"""
    try:
        token = await oauth.github.authorize_access_token(request)
        
        # GitHub doesn't return user info in token, we need to fetch it
        import httpx
        async with httpx.AsyncClient() as client:
            headers = {
                'Authorization': f"Bearer {token['access_token']}",
                'Accept': 'application/json'
            }
            
            # Get user profile
            user_resp = await client.get('https://api.github.com/user', headers=headers)
            user_info = user_resp.json()
            
            # Get primary email (GitHub might not return email in profile)
            email = user_info.get('email')
            if not email:
                emails_resp = await client.get('https://api.github.com/user/emails', headers=headers)
                emails = emails_resp.json()
                primary_email = next((e for e in emails if e.get('primary')), None)
                email = primary_email.get('email') if primary_email else None
            
            if not email:
                raise HTTPException(status_code=400, detail="Email not provided by GitHub")
            
            name = user_info.get('name') or user_info.get('login', email.split('@')[0])
            
            # Find or create user
            result = await db.execute(select(User).where(User.email == email))
            user = result.scalar_one_or_none()
            
            if user is None:
                # Create new user with OAuth
                # OAuth users are automatically verified since the OAuth provider verified the email
                user = User(
                    name=name,
                    email=email,
                    password_hash=secrets.token_urlsafe(32),  # Random hash, won't be used
                    email_verified=True,  # OAuth users are pre-verified
                    verification_token=None
                )
                db.add(user)
                await db.commit()
                await db.refresh(user)
            
            # Generate tokens
            access = create_access_token({"id": user.id, "email": user.email})
            refresh = create_refresh_token({"id": user.id, "email": user.email, "tv": user.token_version})
            
            # Redirect back to frontend with tokens
            frontend_url = settings.frontend_url
            redirect_url = f"{frontend_url}?token={access}&refreshToken={refresh}"
            return RedirectResponse(url=redirect_url)
            
    except OAuthError as e:
        raise HTTPException(status_code=400, detail=f"OAuth error: {str(e)}")
