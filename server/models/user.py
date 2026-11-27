# Each model represents one database entity/table. Keeping them separate makes the application easier to maintain as the number of relationships grows
import uuid

from sqlalchemy import (
    CheckConstraint,
    String,
    UniqueConstraint,
    text,
    func,
)
from sqlalchemy.dialects.mysql import BIGINT, BINARY, DATETIME
from sqlalchemy.orm import Mapped, mapped_column

from server.extensions import db


class User(db.Model):
    __tablename__ = "users"

    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        UniqueConstraint("public_id", name="uq_users_public_id"),
        CheckConstraint(
            "account_status IN ('active', 'suspended', 'pending')",
            name="ck_users_account_status"
        ),
        CheckConstraint(
            "preferred_language IN ('en', 'kn', 'hi')",
            name="ck_users_preferred_language"
        ),
        {"mysql_engine": "InnoDB"}
    )

    # Internal primary key
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )

    # Public UUID (16 bytes)
    public_id: Mapped[bytes] = mapped_column(
        BINARY(16),
        nullable=False,
        default=lambda: uuid.uuid4().bytes
    )

    # User information
    full_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    email: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    # Application preferences
    preferred_language: Mapped[str] = mapped_column(
        String(5),
        nullable=False,
        default="en",
        server_default=text("'en'")
    )

    account_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default=text("'active'")
    )

    # UTC timestamps
    created_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)")
    )

    updated_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)"),
        onupdate=func.current_timestamp(6)
    )
