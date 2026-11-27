from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Shared SQLAlchemy instance
db = SQLAlchemy()

# Database migration manager - will manage schema
migrate = Migrate()

# JWT authentication
jwt = JWTManager()

limiter = Limiter(
    key_func=get_remote_address
)