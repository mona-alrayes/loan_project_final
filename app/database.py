import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Check if running on Render
IS_RENDER = os.getenv('RENDER', 'false').lower() == 'true'

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./loan_app.db")

# Use persistent storage on Render
if IS_RENDER and DATABASE_URL.startswith("sqlite"):
    # Create persistent directory if it doesn't exist
    PERSISTENT_DIR = "/opt/render/project/persistent"
    os.makedirs(PERSISTENT_DIR, exist_ok=True)
    
    # Update database path
    DATABASE_URL = f"sqlite:///{PERSISTENT_DIR}/loan_app.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def create_tables():
    """Create database tables if they don't exist"""
    Base.metadata.create_all(bind=engine)