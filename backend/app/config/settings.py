from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
import os

class Settings(BaseSettings):
    jwt_secret: str = Field(..., alias="JWT_SECRET")
    refresh_secret: str = Field(..., alias="REFRESH_SECRET")
    # Optional session secret for server-side sessions (used by OAuth)
    session_secret: str = Field("", alias="SESSION_SECRET")
    access_expires_hours: int = Field(2, alias="ACCESS_EXPIRES_HOURS")
    refresh_expires_days: int = Field(7, alias="REFRESH_EXPIRES_DAYS")
    database_url: str = Field(..., alias="DATABASE_URL")
    cors_origins: str = Field("http://localhost:5173,http://localhost:5174", alias="CORS_ORIGINS")
    
    # OAuth credentials
    google_client_id: str = Field("", alias="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field("", alias="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field("http://localhost:8000/api/auth/oauth/google/callback", alias="GOOGLE_REDIRECT_URI")
    
    github_client_id: str = Field("", alias="GITHUB_CLIENT_ID")
    github_client_secret: str = Field("", alias="GITHUB_CLIENT_SECRET")
    github_redirect_uri: str = Field("http://localhost:8000/api/auth/oauth/github/callback", alias="GITHUB_REDIRECT_URI")
    
    frontend_url: str = Field("http://localhost:5173", alias="FRONTEND_URL")
    
    # LiveKit configuration
    livekit_api_key: str = Field("", alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field("", alias="LIVEKIT_API_SECRET")
    livekit_url: str = Field("wss://interview-coach-44v9xge4.livekit.cloud", alias="LIVEKIT_URL")

    model_config = ConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
