import re
import hashlib
import secrets
import uuid

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from werkzeug.security import generate_password_hash

from server.extensions import db
from server.models.user import User

from datetime import datetime, timedelta, timezone

from flask_jwt_extended import create_access_token
from werkzeug.security import check_password_hash

from server.models.refresh_token import RefreshToken


# Basic email format validation
EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)

SUPPORTED_LANGUAGES = {"en", "kn", "hi"}

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128


class RegistrationError(Exception):

    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def register_user(data):

    # 1. Validate full name
    full_name = data.get("full_name")

    if not isinstance(full_name, str):
        raise RegistrationError("Full name is required.")

    full_name = full_name.strip()

    if not full_name:
        raise RegistrationError("Full name cannot be empty.")

    if len(full_name) > 100:
        raise RegistrationError("Full name cannot exceed 100 characters.")

    if any(ord(char) < 32 for char in full_name):
        raise RegistrationError("Full name contains invalid characters.")

    # 2. Validate and normalize email
    email = data.get("email")

    if not isinstance(email, str):
        raise RegistrationError("Email is required.")

    email = email.strip().lower()

    if len(email) > 255 or not EMAIL_PATTERN.fullmatch(email):
        raise RegistrationError("Please provide a valid email address.")

    # 3. Validate password
    password = data.get("password")

    if not isinstance(password, str):
        raise RegistrationError("Password is required.")

    if len(password) < MIN_PASSWORD_LENGTH:
        raise RegistrationError(
            f"Password must contain at least {MIN_PASSWORD_LENGTH} characters."
        )

    if len(password) > MAX_PASSWORD_LENGTH:
        raise RegistrationError(
            f"Password cannot exceed {MAX_PASSWORD_LENGTH} characters."
        )

    # 4. Validate preferred language
    preferred_language = data.get("preferred_language", "en")

    if not isinstance(preferred_language, str):
        raise RegistrationError("Invalid preferred language.")

    preferred_language = preferred_language.strip().lower()

    if preferred_language not in SUPPORTED_LANGUAGES:
        raise RegistrationError(
            "Supported languages are: en, kn, hi."
        )

    # 5. Check whether email already exists
    existing_user = db.session.scalar(
        select(User.id).where(User.email == email)
    )

    if existing_user is not None:
        raise RegistrationError(
            "An account with this email already exists.",
            409
        )

    # 6. Hash password
    password_hash = generate_password_hash(
        password,
        method="scrypt"
    )

    # 7. Create user
    user = User(
        full_name=full_name,
        email=email,
        password_hash=password_hash,
        preferred_language=preferred_language
    )

    try:
        db.session.add(user)

        # Commit the user creation as one database transaction
        db.session.commit()

    except IntegrityError as exc:
        db.session.rollback()

        # Handles concurrent requests attempting the same email.
        raise RegistrationError(
            "An account with this email already exists.",
            409
        ) from exc

    except SQLAlchemyError:
        db.session.rollback()
        raise

    return user


class AuthenticationError(Exception):
    def __init__(self, message, status_code=401, code="AUTHENTICATION_ERROR"):
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


ACCESS_TOKEN_MINUTES = 15
REFRESH_TOKEN_DAYS = 30


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _hash_refresh_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _generate_refresh_token():
    return secrets.token_urlsafe(64)


def _create_token_response(user, refresh_token):
    access_token = create_access_token(
        identity=str(uuid.UUID(bytes=user.public_id))
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_MINUTES * 60,
        "refresh_expires_in": REFRESH_TOKEN_DAYS * 24 * 60 * 60
    }


def login_user(data):
    email = data.get("email")
    password = data.get("password")

    if not isinstance(email, str) or not isinstance(password, str):
        raise AuthenticationError("Invalid email or password.")

    email = email.strip().lower()

    user = db.session.scalar(
        select(User).where(User.email == email)
    )

    if user is None or not check_password_hash(
        user.password_hash, password
    ):
        raise AuthenticationError("Invalid email or password.")

    if user.account_status != "active":
        raise AuthenticationError(
            "This account is not available.",
            403,
            "ACCOUNT_UNAVAILABLE"
        )

    raw_refresh_token = _generate_refresh_token()

    refresh_record = RefreshToken(
        user_id=user.id,
        token_hash=_hash_refresh_token(raw_refresh_token),
        token_family_id=uuid.uuid4().bytes,
        expires_at=_utc_now() + timedelta(days=REFRESH_TOKEN_DAYS)
    )

    try:
        db.session.add(refresh_record)
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        raise

    return user, _create_token_response(user, raw_refresh_token)


def refresh_access_token(raw_refresh_token):
    if not isinstance(raw_refresh_token, str) or not raw_refresh_token:
        raise AuthenticationError("Invalid refresh token.")

    token_hash = _hash_refresh_token(raw_refresh_token)

    try:
        token_record = db.session.scalar(
            select(RefreshToken)
            .where(RefreshToken.token_hash == token_hash)
            .with_for_update()
        )

        if token_record is None:
            raise AuthenticationError("Invalid refresh token.")

        now = _utc_now()

        # Detect reuse of a previously revoked refresh token.
        if token_record.revoked_at is not None:
            db.session.execute(
                update(RefreshToken)
                .where(
                    RefreshToken.token_family_id == token_record.token_family_id,
                    RefreshToken.revoked_at.is_(None)
                )
                .values(revoked_at=now)
            )
            db.session.commit()

            raise AuthenticationError(
                "Refresh token reuse detected. Please log in again.",
                401,
                "REFRESH_TOKEN_REUSE"
            )

        if token_record.expires_at <= now:
            token_record.revoked_at = now
            db.session.commit()
            raise AuthenticationError("Refresh token has expired.")

        user = db.session.get(User, token_record.user_id)

        if user is None or user.account_status != "active":
            token_record.revoked_at = now
            db.session.commit()
            raise AuthenticationError("Account is not available.", 403)

        # Revoke the old token before issuing its replacement.
        token_record.revoked_at = now

        new_raw_token = _generate_refresh_token()

        new_record = RefreshToken(
            user_id=user.id,
            token_hash=_hash_refresh_token(new_raw_token),
            token_family_id=token_record.token_family_id,
            expires_at=now + timedelta(days=REFRESH_TOKEN_DAYS)
        )

        db.session.add(new_record)
        db.session.commit()

        return user, _create_token_response(user, new_raw_token)

    except AuthenticationError:
        raise
    except SQLAlchemyError:
        db.session.rollback()
        raise


def logout_user(raw_refresh_token):
    if not isinstance(raw_refresh_token, str) or not raw_refresh_token:
        raise AuthenticationError("Invalid refresh token.")

    token_record = db.session.scalar(
        select(RefreshToken).where(
            RefreshToken.token_hash == _hash_refresh_token(raw_refresh_token)
        )
    )

    if token_record is not None and token_record.revoked_at is None:
        token_record.revoked_at = _utc_now()

        try:
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            raise