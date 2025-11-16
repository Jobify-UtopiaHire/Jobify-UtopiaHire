from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from contextlib import asynccontextmanager
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from .config.settings import settings
from .database.connection import Base, engine
from .routes.auth import router as auth_router
from .routes.oauth import router as oauth_router
from .routes.service import router as service_router
from .routes.job import router as job_router
from .middleware.logging import logging_middleware
from .routes.cv_tools import router as cv_router
from .middleware.security import (
    limiter,
    SecurityHeadersMiddleware,
    RequestSizeLimitMiddleware,
    InputSanitizationMiddleware,
)
# Import Skills Gap Analysis components
import sys
import os
from dotenv import load_dotenv

# Load environment variables first
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)
print(f"🔧 Loading .env from: {env_path}")

# Add parent directory to path to import from core, api, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ⚠️ TEMPORARY: Skills Gap Analysis disabled for faster development
# TODO: Re-enable for production by setting SKILLS_GAP_ENABLED = True
SKILLS_GAP_ENABLED = False

if SKILLS_GAP_ENABLED:
    try:
        from core.skill_extractor import gliner_extractor
        from core.data_loader import data_loader
        from api import analysis
        # Check if Gemini is actually initialized
        from core.ai_analyzer import model as gemini_model
        if gemini_model is None:
            print("⚠️  WARNING: Gemini AI model is not initialized. Skills Gap Analysis will not work.")
            print("    Please check your GOOGLE_API_KEY or GEMINI_API_KEY in .env file")
    except ImportError as e:
        print(f"⚠️  Skills Gap Analysis not available: {e}")
        SKILLS_GAP_ENABLED = False
        gliner_extractor = None
        data_loader = None
        analysis = None
        gemini_model = None
else:
    print("⚠️  Skills Gap Analysis DISABLED for development")
    gliner_extractor = None
    data_loader = None
    analysis = None
    gemini_model = None

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Unified lifespan handler for both database and ML model initialization.
    Replaces deprecated @app.on_event("startup")
    """
    print("=" * 60)
    print("🚀 Starting Unified Backend Service")
    print("=" * 60)
    
    # Initialize database
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Database migrations
            try:
                await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version integer NOT NULL DEFAULT 0"))
            except Exception:
                pass
            try:
                await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified integer NOT NULL DEFAULT 0"))
                await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_token varchar(255)"))
            except Exception:
                pass
            try:
                await conn.execute(text("ALTER TABLE interviews DROP CONSTRAINT IF EXISTS interviews_user_id_fkey"))
                await conn.execute(text("ALTER TABLE interviews ADD CONSTRAINT interviews_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE"))
            except Exception as e:
                print(f"Note: Foreign key constraint update skipped: {e}")
                pass
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
    
    # Initialize Skills Gap Analysis models
    if SKILLS_GAP_ENABLED:
        try:
            print(f"📊 Loaded {len(data_loader.esco_df)} ESCO skills from market data")
            print(f"🤖 GLiNER model '{gliner_extractor.model_name}' loaded and ready")
            
            # Check Gemini model status
            if gemini_model is not None:
                print("✅ Gemini AI model initialized successfully")
            else:
                print("❌ Gemini AI model NOT initialized - Skills Gap Analysis will fail")
                print("   Check your GOOGLE_API_KEY or GEMINI_API_KEY in .env file")
            
            print("✅ Skills Gap Analysis ready")
        except Exception as e:
            print(f"⚠️  Skills Gap Analysis initialization warning: {e}")
    
    print("=" * 60)
    print("📍 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🏥 Health: http://localhost:8000/health")
    print("=" * 60)
    
    yield  # Server is running
    
    # Cleanup on shutdown
    print("🛑 Server shutting down...")

app = FastAPI(
    title="Unified Backend Service - Auth, Services & Skills Gap Analysis",
    description="Complete backend API for authentication, services, jobs, CV tools, and AI-powered skills gap analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Custom validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {exc.errors()}")
    print(f"Request body: {await request.body()}")
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    field = first_error.get('loc', ['unknown'])[-1]
    msg = first_error.get('msg', 'Invalid input')
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": f"Invalid {field}: {msg}"}
    )

# CORS
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Middleware - order matters!
# 1. Security headers (first to apply to all responses)
app.add_middleware(SecurityHeadersMiddleware)

# 2. Request size limits (prevent DoS)
app.add_middleware(RequestSizeLimitMiddleware)

# 3. Input sanitization
app.add_middleware(InputSanitizationMiddleware)

# Logging middleware
app.middleware("http")(logging_middleware)

# Session middleware required for OAuth state management
app.add_middleware(
    SessionMiddleware,
    secret_key=(settings.session_secret or settings.jwt_secret),
    same_site="lax",
    
)

@app.get("/")
async def root():
    """Root endpoint with service information"""
    services = {
        "authentication": "✅ Available",
        "oauth": "✅ Available",
        "services": "✅ Available",
        "jobs": "✅ Available",
        "cv_tools": "✅ Available",
        "skills_gap_analysis": "✅ Available" if SKILLS_GAP_ENABLED else "❌ Not Available"
    }
    return {
        "message": "Welcome to the Unified Backend API",
        "version": "1.0.0",
        "services": services,
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health():
    """Comprehensive health check for all services"""
    health_status = {
        "status": "ok",
        "timestamp": None,
        "services": {}
    }
    
    # Check database
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health_status["services"]["database"] = "up"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["services"]["database"] = f"down: {str(e)}"
    
    # Check Skills Gap Analysis
    if SKILLS_GAP_ENABLED:
        try:
            skills_available = data_loader is not None and len(data_loader.esco_df) > 0
            model_loaded = gliner_extractor is not None
            health_status["services"]["skills_gap"] = {
                "status": "up" if (skills_available and model_loaded) else "degraded",
                "esco_skills_loaded": len(data_loader.esco_df) if data_loader else 0,
                "model_loaded": model_loaded
            }
        except Exception as e:
            health_status["services"]["skills_gap"] = f"error: {str(e)}"
    else:
        health_status["services"]["skills_gap"] = "not_enabled"
    
    # Add timestamp
    from datetime import datetime
    health_status["timestamp"] = datetime.utcnow().isoformat()
    
    return health_status

# Include routers
app.include_router(auth_router, tags=["Authentication"])
app.include_router(oauth_router, tags=["OAuth"])
app.include_router(service_router, tags=["Services"])
app.include_router(job_router, tags=["Jobs"])
app.include_router(cv_router, tags=["CvTools"])

# Include Skills Gap Analysis router if available
if SKILLS_GAP_ENABLED and analysis:
    app.include_router(analysis.router, prefix="/api", tags=["Skills Gap Analysis"])
    print("✅ Skills Gap Analysis routes registered at /api/analyze")
else:
    print("⚠️  Skills Gap Analysis routes not available")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=["app", "core", "api", "cvtool", "agent"]
    )