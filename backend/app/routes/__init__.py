"""
API routes module
"""
from .auth import router as auth_router
from .oauth import router as oauth_router
from .service import router as service_router
from .job import router as job_router
from .cv_tools import router as cv_router

__all__ = [
    "auth_router",
    "oauth_router",
    "service_router",
    "job_router",
    "cv_router"
]
