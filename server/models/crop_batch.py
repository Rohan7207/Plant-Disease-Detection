
import uuid

from sqlalchemy import (
    String,
    Text,
    Date,
    Numeric,
    ForeignKey,
    UniqueConstraint,
    Index,
    CheckConstraint,
    text,
)
from sqlalchemy.dialects.mysql import BIGINT, BINARY, INTEGER, DATETIME
from sqlalchemy.orm import Mapped, mapped_column

from server.extensions import db


class CropBatch(db.Model):
    __tablename__ = "crop_batches"

    __table_args__ = (
        UniqueConstraint("public_id", name="uq_crop_batches_public_id"),
        UniqueConstraint(
            "id", "user_id",
            name="uq_crop_batches_id_user"
        ),
        Index(
            "ix_crop_batches_user_deleted",
            "user_id",
            "deleted_at"
        ),
        CheckConstraint(
            "cultivated_area IS NULL OR cultivated_area > 0",
            name="ck_crop_batches_positive_area"
        ),
        CheckConstraint(
            "plant_count IS NULL OR plant_count > 0",
            name="ck_crop_batches_positive_plant_count"
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

    # Ownership
    user_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False
    )

    # Crop relationship
    crop_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("crops.id", ondelete="RESTRICT"),
        nullable=False
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    variety: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    planted_on: Mapped[object | None] = mapped_column(
        Date,
        nullable=True
    )

    cultivated_area: Mapped[object | None] = mapped_column(
        Numeric(12, 4),
        nullable=True
    )

    area_unit: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True
    )

    plant_count: Mapped[int | None] = mapped_column(
        INTEGER(unsigned=True),
        nullable=True
    )

    location: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    created_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)")
    )

    updated_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)"),
        onupdate=text("CURRENT_TIMESTAMP(6)")
    )

    deleted_at: Mapped[object | None] = mapped_column(
        DATETIME(fsp=6),
        nullable=True
    )
