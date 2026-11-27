
from sqlalchemy import String, Boolean, UniqueConstraint, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME
from sqlalchemy.orm import Mapped, mapped_column

from server.extensions import db


class Crop(db.Model):
    __tablename__ = "crops"

    __table_args__ = (
        UniqueConstraint("code", name="uq_crops_code"),
        {"mysql_engine": "InnoDB"}
    )

    # Internal primary key
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )

    # Crop identification
    code: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    display_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    scientific_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True
    )

    # Availability of Crop
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("1")
    )

    # Creation timestamp
    created_at: Mapped[object] = mapped_column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)")
    )
