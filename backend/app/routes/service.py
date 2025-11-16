import os
import secrets
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from livekit import api
from ..config.settings import settings
from ..database.connection import get_db
from ..models.user import User, Interview
from ..schemas.auth_schemas import StartSessionRequest, SaveInterviewRequest
from ..utils.security_utils import get_current_user
from ..middleware.security import (
    limiter,
    validate_file_upload,
    get_safe_filename,
)

router = APIRouter(tags=["services"])

# Upload configuration
UPLOAD_FOLDER = 'cv_uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Session data storage - shared with agent.py via import
session_data = {}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@router.get('/health')
async def health():
    return {
        "status": "ok", 
        "service": "Unified Backend Service"
    }

@router.post('/api/upload-cv')
@limiter.limit("10/minute")
async def upload_cv(
    request: Request,
    cv: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload CV file - JWT protected with rate limiting"""
    try:
        # Check if file is provided
        if not cv:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if cv.filename == '':
            raise HTTPException(status_code=400, detail="No file selected")
        
        # Read file content
        file_content = await cv.read()
        
        # Validate file using security middleware
        is_valid, error_message = validate_file_upload(cv.filename, file_content)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Generate secure filename
        safe_filename = get_safe_filename(cv.filename)
        filename = f"{secrets.token_hex(8)}_{safe_filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save file securely
        with open(filepath, 'wb') as f:
            f.write(file_content)
        
        # Set restrictive permissions (owner read/write only)
        os.chmod(filepath, 0o600)
        
        print(f"✅ CV uploaded by user {current_user.email}: {filename}")
        
        return {
            "success": True,
            "message": "CV uploaded successfully",
            "filename": filename,
            "path": filepath
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error uploading CV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post('/api/start-session')
@limiter.limit("20/minute")
async def start_session(
    request: Request,
    data: StartSessionRequest,
    current_user: User = Depends(get_current_user)
):
    """Start interview session - JWT protected"""
    try:
        if not settings.livekit_api_key or not settings.livekit_api_secret:
            raise HTTPException(status_code=500, detail="LiveKit credentials not configured")
        
        identity = data.identity or f"user-{secrets.token_hex(4)}"
        name = data.name or 'Interview Candidate'
        room_name = data.room or f"interview-{secrets.token_hex(4)}"
        cv_filename = data.cv_filename
        job_description = data.job_description
        candidate_email = data.candidate_email
        candidate_name = data.candidate_name
        job_title = data.job_title
        language = data.language
        
        # Normalize provided cv filename: ensure it maps to an actual saved file in UPLOAD_FOLDER
        candidate_path = os.path.join(UPLOAD_FOLDER, cv_filename)
        print(f"🔍 Looking for CV file: {cv_filename}")
        print(f"📁 Full path: {candidate_path}")
        
        if not os.path.exists(candidate_path):
            print(f"⚠️ File not found at path, trying to match...")
            print(f"📂 Files in upload folder: {os.listdir(UPLOAD_FOLDER)}")
            
            # Try multiple matching strategies
            matched = None
            
            # Strategy 1: Try to find a saved file that ends with the provided filename
            for f in os.listdir(UPLOAD_FOLDER):
                if f.endswith(cv_filename):
                    matched = f
                    print(f"✅ Strategy 1 - Matched by suffix: {matched}")
                    break
            
            # Strategy 2: If still not found, try sanitized filename
            if not matched:
                sanitized = "".join(c for c in cv_filename if c.isalnum() or c in '._- ')
                for f in os.listdir(UPLOAD_FOLDER):
                    if f.endswith(sanitized):
                        matched = f
                        print(f"✅ Strategy 2 - Matched by sanitized suffix: {matched}")
                        break
            
            # Strategy 3: If still not found, try matching by base filename (without hash prefix)
            if not matched:
                # Extract the base name without the hash prefix
                base_name = cv_filename.split('_', 1)[-1] if '_' in cv_filename else cv_filename
                for f in os.listdir(UPLOAD_FOLDER):
                    if f.endswith(base_name):
                        matched = f
                        print(f"✅ Strategy 3 - Matched by base name: {matched}")
                        break
            
            if matched:
                cv_filename = matched
                print(f"🔎 Final mapped filename: {cv_filename}")
            else:
                print(f"❌ No match found for: {cv_filename}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"CV file not found on server: {cv_filename}. Available files: {', '.join(os.listdir(UPLOAD_FOLDER))}"
                )

        # Store session data for agent to retrieve (use the actual saved filename)
        session_data[room_name] = {
            "cv_filename": cv_filename,
            "job_description": job_description,
            "candidate_email": candidate_email,
            "candidate_name": candidate_name,
            "job_title": job_title,
            "language": language
        }
        print(f"📝 Session data stored for room: {room_name}")
        print(f"👤 Candidate: {candidate_name} ({candidate_email})")
        print(f"💼 Position: {job_title}")
        print(f"🔐 Initiated by user: {current_user.email}")
        
        # Create access token
        token = api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        token.with_identity(identity)
        token.with_name(name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
        ))
        
        jwt_token = token.to_jwt()
        
        response_data = {
            "token": jwt_token,
            "url": settings.livekit_url,
            "identity": identity,
            "room": room_name
        }
        
        print(f"✅ Session started for {identity} in room {room_name}")
        print(f"📄 CV: {cv_filename}")
        print(f"💼 Job: {job_description[:50]}...")
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start session")

@router.get('/api/session-data/{room_name}')
async def get_session_data(room_name: str):
    """Get session data for a room (used by agent) - No JWT protection for agent access"""
    if room_name in session_data:
        return session_data[room_name]
    raise HTTPException(status_code=404, detail="Session data not found")

@router.post("/api/interviews/save-report")
async def save_interview_report(
    data: SaveInterviewRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Save interview report to database after email is sent
    No JWT protection - called by agent after interview
    """
    try:
        # Find user by candidate_email
        result = await db.execute(select(User).where(User.email == data.candidate_email))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User not found with email: {data.candidate_email}"
            )
        
        # Create new interview record with existing schema
        interview = Interview(
            user_id=user.id,
            job_title=data.job_title,
            interview_score=data.interview_score,
            conclusion=data.conclusion
        )
        
        db.add(interview)
        await db.commit()
        await db.refresh(interview)
        
        print(f"✅ Interview saved to database - ID: {interview.id}, User ID: {user.id}")
        return {
            "success": True,
            "interview_id": interview.id,
            "user_id": user.id,
            "message": "Interview record created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error saving interview: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/api/interviews/email/{email}')
async def get_user_interviews_by_email(
    email: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all interviews for a user by their email - JWT protected
    Users can only access their own interviews
    """
    try:
        # Security: Users can only access their own interviews
        if current_user.email != email:
            raise HTTPException(status_code=403, detail="You can only access your own interviews")
        
        # Get all interviews for this user
        result = await db.execute(select(Interview).where(Interview.user_id == current_user.id))
        interviews = result.scalars().all()
        
        # Format response
        interviews_data = [
            {
                "id": interview.id,
                "job_title": interview.job_title,
                "interview_score": interview.interview_score,
                "conclusion": interview.conclusion,
                "created_at": interview.created_at.isoformat() if interview.created_at else None
            }
            for interview in interviews
        ]
        
        print(f"✅ Retrieved {len(interviews_data)} interviews for user {current_user.name}")      
        return {
            "success": True,
            "user_id": current_user.id,
            "user_name": current_user.name,
            "user_email": current_user.email,
            "total_interviews": len(interviews_data),
            "interviews": interviews_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error retrieving interviews: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve interviews: {str(e)}")
