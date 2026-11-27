import uuid

from sqlalchemy import select

from server.extensions import db
from server.extensions import limiter
from server.models.user import User

from flask import Blueprint, request, jsonify, current_app
from sqlalchemy.exc import SQLAlchemyError
from flask_jwt_extended import jwt_required, get_jwt_identity

from server.services.auth_service import (
    register_user,
    RegistrationError
)

from server.services.auth_service import (
    login_user,
    refresh_access_token,
    logout_user,
    AuthenticationError
)


auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["POST"])
def register():

    # 1. Require JSON request
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": {
                "code": "UNSUPPORTED_MEDIA_TYPE",
                "message": "Content-Type must be application/json."
            }
        }), 415

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": {
                "code": "INVALID_REQUEST",
                "message": "A valid JSON object is required."
            }
        }), 400

    try:
        # 2. Call registration service
        user = register_user(data)

        # 3. Return safe user information
        return jsonify({
            "success": True,
            "message": "Account registered successfully.",
            "data": {
                "user": {
                   "public_id": str(uuid.UUID(bytes=user.public_id)),
                    "full_name": user.full_name,
                    "email": user.email,
                    "preferred_language": user.preferred_language,
                    "created_at": user.created_at.isoformat()
                }
            }
        }), 201

    except RegistrationError as exc:

        return jsonify({
            "success": False,
            "error": {
                "code": (
                    "EMAIL_ALREADY_EXISTS"
                    if exc.status_code == 409
                    else "VALIDATION_ERROR"
                ),
                "message": exc.message
            }
        }), exc.status_code

    except SQLAlchemyError:

        current_app.logger.exception(
            "Database error during user registration."
        )

        return jsonify({
            "success": False,
            "error": {
                "code": "DATABASE_ERROR",
                "message": "Unable to register account. Please try again."
            }
        }), 500
        
@auth_bp.route("/login", methods=["POST"])
@limiter.limit("5 per minute")
def login():
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": {
                "code": "UNSUPPORTED_MEDIA_TYPE",
                "message": "Content-Type must be application/json."
            }
        }), 415

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": {
                "code": "INVALID_REQUEST",
                "message": "A valid JSON object is required."
            }
        }), 400

    try:
        user, tokens = login_user(data)

        return jsonify({
            "success": True,
            "message": "Login successful.",
            "data": {
                "user": {
                    "public_id": str(uuid.UUID(bytes=user.public_id)),
                    "full_name": user.full_name,
                    "email": user.email,
                    "preferred_language": user.preferred_language
                },
                "tokens": tokens
            }
        }), 200

    except AuthenticationError as exc:
        return jsonify({
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message
            }
        }), exc.status_code

    except SQLAlchemyError:
        current_app.logger.exception("Database error during login.")
        return jsonify({
            "success": False,
            "error": {
                "code": "DATABASE_ERROR",
                "message": "Unable to process login. Please try again."
            }
        }), 500
        
@auth_bp.route("/refresh", methods=["POST"])
def refresh():
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": {
                "code": "UNSUPPORTED_MEDIA_TYPE",
                "message": "Content-Type must be application/json."
            }
        }), 415

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": {
                "code": "INVALID_REQUEST",
                "message": "A valid JSON object is required."
            }
        }), 400

    try:
        user, tokens = refresh_access_token(
            data.get("refresh_token")
        )

        return jsonify({
            "success": True,
            "message": "Tokens refreshed successfully.",
            "data": {
                "user": {
                    "public_id": str(uuid.UUID(bytes=user.public_id)),
                    "email": user.email
                },
                "tokens": tokens
            }
        }), 200

    except AuthenticationError as exc:
        return jsonify({
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message
            }
        }), exc.status_code

    except SQLAlchemyError:
        current_app.logger.exception("Database error during token refresh.")
        return jsonify({
            "success": False,
            "error": {
                "code": "DATABASE_ERROR",
                "message": "Unable to refresh tokens. Please try again."
            }
        }), 500
        
@auth_bp.route("/logout", methods=["POST"])
def logout():
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": {
                "code": "UNSUPPORTED_MEDIA_TYPE",
                "message": "Content-Type must be application/json."
            }
        }), 415

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": {
                "code": "INVALID_REQUEST",
                "message": "A valid JSON object is required."
            }
        }), 400

    try:
        logout_user(data.get("refresh_token"))

        return jsonify({
            "success": True,
            "message": "Logout successful."
        }), 200

    except AuthenticationError as exc:
        return jsonify({
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message
            }
        }), exc.status_code

    except SQLAlchemyError:
        current_app.logger.exception("Database error during logout.")
        return jsonify({
            "success": False,
            "error": {
                "code": "DATABASE_ERROR",
                "message": "Unable to process logout. Please try again."
            }
        }), 500
        
@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def get_current_user():
    public_id = get_jwt_identity()

    try:
        user = db.session.scalar(
            select(User).where(
                User.public_id == uuid.UUID(public_id).bytes
            )
        )

        if user is None or user.account_status != "active":
            return jsonify({
                "success": False,
                "error": {
                    "code": "USER_UNAVAILABLE",
                    "message": "User account is unavailable."
                }
            }), 403

        return jsonify({
            "success": True,
            "data": {
                "user": {
                    "public_id": public_id,
                    "full_name": user.full_name,
                    "email": user.email,
                    "preferred_language": user.preferred_language
                }
            }
        }), 200

    except SQLAlchemyError:
        current_app.logger.exception(
            "Database error while retrieving current user."
        )
        return jsonify({
            "success": False,
            "error": {
                "code": "DATABASE_ERROR",
                "message": "Unable to retrieve user information."
            }
        }), 500