
from sqlalchemy import (
    String,
    ForeignKey,
    UniqueConstraint,
    Index,
    CheckConstraint,
    text,
)
from sqlalchemy.dialects.mysql import BIGINT, DATETIME
from sqlalchemy.orm import Mapped, mapped_column

from server.extensions import db


class ImageAsset(db.Model):
    __tablename__ = "image_assets"

    __table_args__ = (
        UniqueConstraint(
            "storage_provider",
            "storage_key",
            name="uq_image_assets_provider_storage_key"
        ),
        UniqueConstraint(
            "id",
            "user_id",
            name="uq_image_assets_id_user"
        ),
        Index(
            "ix_image_assets_user_created",
            "user_id",
            "created_at"
        ),
        CheckConstraint(
            "file_size_bytes > 0",
            name="ck_image_assets_positive_file_size"
        ),
        {"mysql_engine": "InnoDB"}
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )

    # Image owner
    user_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False
    )
    
    storage_provider: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="imagekit",
        server_default=text("'imagekit'")
    )

    # Storage information
    storage_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    provider_file_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    original_filename: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    mime_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    file_size_bytes: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False
    )

    sha256_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False
    )

    created_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)")
    )
