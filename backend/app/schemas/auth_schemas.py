from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    refreshToken: str = Field(..., alias="refreshToken")

class VerifyEmailRequest(BaseModel):
    token: str

class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr

class TokenResponse(BaseModel):
    token: str
    refreshToken: str
    user: UserOut

class MeResponse(BaseModel):
    user: UserOut

class Message(BaseModel):
    message: str

# Service schemas
class StartSessionRequest(BaseModel):
    identity: Optional[str] = None
    name: Optional[str] = "Interview Candidate"
    room: Optional[str] = None
    cv_filename: str
    job_description: str
    candidate_email: EmailStr
    candidate_name: str
    job_title: str
    language: str

class SaveInterviewRequest(BaseModel):
    candidate_email: EmailStr
    job_title: str
    interview_score: Optional[float] = None
    conclusion: Optional[str] = None
