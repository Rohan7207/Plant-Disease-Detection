
import uuid

from sqlalchemy import (
    String,
    Text,
    Date,
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


class Treatment(db.Model):
    __tablename__ = "treatments"

    __table_args__ = (
        UniqueConstraint(
            "public_id",
            name="uq_treatments_public_id"
        ),

        ForeignKeyConstraint(
            ["plant_id", "user_id"],
            ["plants.id", "plants.user_id"],
            name="fk_treatments_plant_owner",
            ondelete="RESTRICT"
        ),

        ForeignKeyConstraint(
            ["batch_id", "user_id"],
            ["crop_batches.id", "crop_batches.user_id"],
            name="fk_treatments_batch_owner",
            ondelete="RESTRICT"
        ),

        CheckConstraint(
            """
            (plant_id IS NOT NULL AND batch_id IS NULL)
            OR
            (plant_id IS NULL AND batch_id IS NOT NULL)
            """,
            name="ck_treatments_single_parent"
        ),

        CheckConstraint(
            "status IN ('planned', 'in_progress', 'completed', 'cancelled')",
            name="ck_treatments_status"
        ),

        Index(
            "ix_treatments_plant_date",
            "plant_id",
            "treatment_date"
        ),

        Index(
            "ix_treatments_batch_date",
            "batch_id",
            "treatment_date"
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

    treatment_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    treatment_date: Mapped[object] = mapped_column(
        Date,
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="planned",
        server_default=text("'planned'")
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
