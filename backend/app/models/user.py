from sqlalchemy.orm import Mapped, mapped_column , relationship
from sqlalchemy.sql import func
from sqlalchemy import String, Integer, Text, DateTime, ForeignKey, Float
from ..database.connection import Base
from typing import List 

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    email_verified: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)
    verification_token: Mapped[str] = mapped_column(String(255), nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    interviews: Mapped[List["Interview"]] = relationship("Interview", back_populates="user", cascade="all, delete-orphan")

class Interview(Base):
    __tablename__ = "interviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    job_title: Mapped[str] = mapped_column(Text, nullable=False)
    interview_score: Mapped[float] = mapped_column(Float, nullable=True)
    conclusion: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user: Mapped["User"] = relationship("User", back_populates="interviews")
