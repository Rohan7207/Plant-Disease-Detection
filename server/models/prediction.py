
import uuid

from sqlalchemy import (
    String,
    ForeignKey,
    ForeignKeyConstraint,
    UniqueConstraint,
    Index,
    CheckConstraint,
    Numeric,
    text,
)
from sqlalchemy.dialects.mysql import BIGINT, BINARY, DATETIME
from sqlalchemy.orm import Mapped, mapped_column

from server.extensions import db


class Prediction(db.Model):
    __tablename__ = "predictions"

    __table_args__ = (
        UniqueConstraint(
            "public_id",
            name="uq_predictions_public_id"
        ),

        # Enforce that the image belongs to the same user.
        ForeignKeyConstraint(
            ["image_asset_id", "user_id"],
            ["image_assets.id", "image_assets.user_id"],
            name="fk_predictions_image_owner",
            ondelete="RESTRICT"
        ),

        # Enforce plant ownership.
        ForeignKeyConstraint(
            ["plant_id", "user_id"],
            ["plants.id", "plants.user_id"],
            name="fk_predictions_plant_owner",
            ondelete="RESTRICT"
        ),

        # Enforce batch ownership.
        ForeignKeyConstraint(
            ["batch_id", "user_id"],
            ["crop_batches.id", "crop_batches.user_id"],
            name="fk_predictions_batch_owner",
            ondelete="RESTRICT"
        ),

        # Ensure the predicted disease belongs to the detected crop.
        ForeignKeyConstraint(
            ["crop_id", "disease_id"],
            ["diseases.crop_id", "diseases.id"],
            name="fk_predictions_crop_disease",
            ondelete="RESTRICT"
        ),

        # Crop must exist even when disease_id is NULL.
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1",
            name="ck_predictions_confidence_range"
        ),

        # A prediction may reference a plant, a batch, or neither.
        # It must never reference both.
        CheckConstraint(
            "plant_id IS NULL OR batch_id IS NULL",
            name="ck_predictions_single_parent"
        ),

        Index(
            "ix_predictions_user_created",
            "user_id",
            "created_at"
        ),

        Index(
            "ix_predictions_plant_created",
            "plant_id",
            "created_at"
        ),

        Index(
            "ix_predictions_batch_created",
            "batch_id",
            "created_at"
        ),

        {"mysql_engine": "InnoDB"}
    )

    # Internal primary key
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )

    # Public UUID
    public_id: Mapped[bytes] = mapped_column(
        BINARY(16),
        nullable=False,
        default=lambda: uuid.uuid4().bytes
    )

    # User who performed the prediction
    user_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False
    )

    # Optional individual plant association
    plant_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        nullable=True
    )

    # Optional crop batch association
    batch_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        nullable=True
    )

    # Uploaded image reference
    image_asset_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False
    )

    # Predicted crop
    crop_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("crops.id", ondelete="RESTRICT"),
        nullable=False
    )

    # Predicted disease (NULL for healthy/unclassified results)
    disease_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        nullable=True
    )

    # AI confidence score
    confidence_score: Mapped[object] = mapped_column(
        Numeric(6, 5),
        nullable=False
    )

    # AI model version
    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    # Prediction timestamp
    created_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)")
    )
