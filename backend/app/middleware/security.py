"""
Comprehensive Security Middleware for FastAPI Application
Implements multiple layers of security including rate limiting, headers, input validation
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import re
import bleach
from typing import Optional, Dict, Any
import time
from collections import defaultdict
from datetime import datetime, timedelta

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Track failed login attempts
failed_login_attempts: Dict[str, list] = defaultdict(list)
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION = 15  # minutes

# Request size limits
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
MAX_JSON_SIZE = 1 * 1024 * 1024  # 1MB for JSON


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds comprehensive security headers to all responses
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy - Allow HTTP for development
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' http://localhost:* http://127.0.0.1:* https: wss:; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp
        
        # Remove server header information
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Limits the size of incoming requests to prevent DoS attacks
    """
    async def dispatch(self, request: Request, call_next):
        # Check Content-Length header
        content_length = request.headers.get("content-length")
        
        if content_length:
            content_length = int(content_length)
            
            # Different limits for different content types
            content_type = request.headers.get("content-type", "")
            
            if "multipart/form-data" in content_type or "application/octet-stream" in content_type:
                # File upload - allow larger size
                if content_length > MAX_REQUEST_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large. Maximum size is 10MB"}
                    )
            elif "application/json" in content_type:
                # JSON requests - smaller limit
                if content_length > MAX_JSON_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "JSON payload too large. Maximum size is 1MB"}
                    )
        
        response = await call_next(request)
        return response


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """
    Sanitizes inputs to prevent XSS and injection attacks
    """
    async def dispatch(self, request: Request, call_next):
        # For now, just pass through - sanitization happens at validation layer
        # This middleware serves as a placeholder for future enhancements
        response = await call_next(request)
        return response


def sanitize_html(text: str) -> str:
    """
    Sanitize HTML input to prevent XSS attacks
    """
    if not text:
        return text
    
    # Allow no HTML tags - strip everything
    allowed_tags = []
    allowed_attrs = {}
    
    cleaned = bleach.clean(
        text,
        tags=allowed_tags,
        attributes=allowed_attrs,
        strip=True
    )
    
    return cleaned


def sanitize_sql_input(text: str) -> str:
    """
    Basic SQL injection pattern detection
    Note: This is a secondary defense - primary defense is parameterized queries
    """
    if not text:
        return text
    
    # Dangerous SQL patterns
    dangerous_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|#|\/\*|\*\/)",
        r"(\bOR\b.*=.*)",
        r"(\bAND\b.*=.*)",
        r"(;.*--)",
        r"(UNION.*SELECT)",
        r"(XP_CMDSHELL)",
    ]
    
    text_upper = text.upper()
    for pattern in dangerous_patterns:
        if re.search(pattern, text_upper, re.IGNORECASE):
            raise HTTPException(
                status_code=400,
                detail="Invalid input detected. Please remove special SQL characters."
            )
    
    return text


def validate_email_format(email: str) -> bool:
    """
    Validate email format
    """
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_pattern.match(email))


def is_account_locked(email: str) -> bool:
    """
    Check if account is locked due to too many failed login attempts
    """
    if email not in failed_login_attempts:
        return False
    
    attempts = failed_login_attempts[email]
    
    # Clean old attempts (older than lockout duration)
    cutoff_time = datetime.now() - timedelta(minutes=LOCKOUT_DURATION)
    attempts[:] = [attempt_time for attempt_time in attempts if attempt_time > cutoff_time]
    
    # Check if account should be locked
    if len(attempts) >= MAX_FAILED_ATTEMPTS:
        return True
    
    return False


def record_failed_login(email: str):
    """
    Record a failed login attempt
    """
    failed_login_attempts[email].append(datetime.now())


def clear_failed_logins(email: str):
    """
    Clear failed login attempts for successful login
    """
    if email in failed_login_attempts:
        failed_login_attempts[email].clear()


def validate_file_upload(filename: str, content: bytes) -> tuple[bool, Optional[str]]:
    """
    Validate file uploads for security
    Returns: (is_valid, error_message)
    """
    # Check filename
    if not filename:
        return False, "Filename is required"
    
    # Check for directory traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        return False, "Invalid filename - directory traversal detected"
    
    # Check file extension
    allowed_extensions = ['pdf', 'doc', 'docx', 'txt', 'odt', 'tex', 'html', 'rtf']
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    if ext not in allowed_extensions:
        return False, f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
    
    # Check file size
    max_size = 10 * 1024 * 1024  # 10MB
    if len(content) > max_size:
        return False, "File size exceeds 10MB limit"
    
    # Check for executable content in filename
    dangerous_extensions = ['exe', 'bat', 'cmd', 'sh', 'ps1', 'vbs', 'js', 'jar']
    if ext in dangerous_extensions:
        return False, "Executable files are not allowed"
    
    # Basic magic number checks for common file types
    # PDF should start with %PDF
    if ext == 'pdf' and not content.startswith(b'%PDF'):
        return False, "File does not appear to be a valid PDF"
    
    # DOCX is a ZIP archive (PK)
    if ext == 'docx' and not content.startswith(b'PK'):
        return False, "File does not appear to be a valid DOCX"
    
    # DOC files start with specific headers
    if ext == 'doc' and not (content.startswith(b'\xD0\xCF\x11\xE0') or content.startswith(b'\xDB\xA5')):
        return False, "File does not appear to be a valid DOC"
    
    return True, None


def get_safe_filename(filename: str) -> str:
    """
    Generate a safe filename by removing dangerous characters
    """
    # Remove directory components
    filename = filename.replace('/', '').replace('\\', '').replace('..', '')
    
    # Keep only alphanumeric, dots, hyphens, underscores
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Ensure it's not empty
    if not safe_name:
        safe_name = "file"
    
    return safe_name


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent abuse
    """
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = get_remote_address(request)
        now = time.time()
        
        # Clean old requests
        cutoff = now - self.period
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > cutoff
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.calls:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Maximum {self.calls} requests per {self.period} seconds."
                }
            )
        
        # Record this request
        self.requests[client_ip].append(now)
        
        response = await call_next(request)
        return response


# Export rate limiter for use in routes
__all__ = [
    'limiter',
    'SecurityHeadersMiddleware',
    'RequestSizeLimitMiddleware',
    'InputSanitizationMiddleware',
    'RateLimitMiddleware',
    'sanitize_html',
    'sanitize_sql_input',
    'validate_email_format',
    'is_account_locked',
    'record_failed_login',
    'clear_failed_logins',
    'validate_file_upload',
    'get_safe_filename',
]
