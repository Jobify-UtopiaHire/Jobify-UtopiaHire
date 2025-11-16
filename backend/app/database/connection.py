from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from ..config.settings import settings

class Base(DeclarativeBase):
    pass

engine: AsyncEngine = create_async_engine(settings.database_url, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
