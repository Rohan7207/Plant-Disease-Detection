
import uuid

from sqlalchemy import (
    String,
    Text,
    ForeignKey,
    ForeignKeyConstraint,
    UniqueConstraint,
    Index,
    CheckConstraint,
    text,
)
from sqlalchemy.dialects.mysql import BIGINT, BINARY, DATETIME
from sqlalchemy.orm import Mapped, mapped_column

from server.extensions import db


class PlantObservation(db.Model):
    __tablename__ = "plant_observations"

    __table_args__ = (
        UniqueConstraint(
            "public_id",
            name="uq_plant_observations_public_id"
        ),

        # Ensure the plant belongs to the same user.
        ForeignKeyConstraint(
            ["plant_id", "user_id"],
            ["plants.id", "plants.user_id"],
            name="fk_observations_plant_owner",
            ondelete="RESTRICT"
        ),

        # Ensure the batch belongs to the same user.
        ForeignKeyConstraint(
            ["batch_id", "user_id"],
            ["crop_batches.id", "crop_batches.user_id"],
            name="fk_observations_batch_owner",
            ondelete="RESTRICT"
        ),

        # An observation must belong to exactly one plant or batch.
        CheckConstraint(
            "(plant_id IS NOT NULL AND batch_id IS NULL) OR "
            "(plant_id IS NULL AND batch_id IS NOT NULL)",
            name="ck_observations_single_parent"
        ),

        Index(
            "ix_observations_plant_date",
            "plant_id",
            "observed_at"
        ),

        Index(
            "ix_observations_batch_date",
            "batch_id",
            "observed_at"
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

    plant_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        nullable=True
    )

    batch_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        nullable=True
    )

    observation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    observed_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)")
    )

    created_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)")
    )
