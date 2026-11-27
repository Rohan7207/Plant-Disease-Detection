
import uuid

from sqlalchemy import (
    String,
    ForeignKey,
    UniqueConstraint,
    Index,
    text,
)
from sqlalchemy.dialects.mysql import BIGINT, BINARY, DATETIME
from sqlalchemy.orm import Mapped, mapped_column

from server.extensions import db


class RefreshToken(db.Model):
    __tablename__ = "refresh_tokens"

    __table_args__ = (
        UniqueConstraint(
            "public_id",
            name="uq_refresh_tokens_public_id"
        ),

        UniqueConstraint(
            "token_hash",
            name="uq_refresh_tokens_token_hash"
        ),

        Index(
            "ix_refresh_tokens_user_expires",
            "user_id",
            "expires_at"
        ),

        Index(
            "ix_refresh_tokens_family",
            "token_family_id"
        ),

        {"mysql_engine": "InnoDB"}
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )

    public_id: Mapped[bytes] = mapped_column(
        BINARY(16),
        nullable=False,
        default=lambda: uuid.uuid4().bytes
    )

    user_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False
    )

    token_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    token_family_id: Mapped[bytes] = mapped_column(
        BINARY(16),
        nullable=False,
        default=lambda: uuid.uuid4().bytes
    )

    expires_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False
    )

    revoked_at: Mapped[object | None] = mapped_column(
        DATETIME(fsp=6),
        nullable=True
    )

    created_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)")
    )
