
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    # Flask security
    SECRET_KEY = os.getenv("SECRET_KEY")

    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY is missing from the .env file.")

    # MySQL configuration
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_NAME = os.getenv("DB_NAME", "plant_disease_db")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://"
        f"{quote_plus(DB_USER)}:"
        f"{quote_plus(DB_PASSWORD)}@"
        f"{DB_HOST}:{DB_PORT}/"
        f"{DB_NAME}?charset=utf8mb4"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Database connection pool
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,  # Check whether database connection is alive before SQLAlchemy reuses it
        "pool_recycle": 1800,
        "pool_size": 5,
        "max_overflow": 10,
        "connect_args": {
            "connect_timeout": 10
        }
    }
    
    # JWT configuration
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = 15 * 60       # 15 minutes
    JWT_REFRESH_TOKEN_EXPIRES = 30 * 24 * 60 * 60  # 30 days
    JWT_TOKEN_LOCATION = ["headers"]

    # File upload configuration
    MAX_CONTENT_LENGTH = int(
        os.getenv("MAX_CONTENT_LENGTH", 10 * 1024 * 1024)
    )

    # Application configuration
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:8081"
        ).split(",")
        if origin.strip()
    ]

    JSON_SORT_KEYS = False
