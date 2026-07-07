"""Add picks.closing_point for spread and total line drift.

Revision ID: b7d8e9f0a1c2
Revises: a1b2c3d4e5f6
Create Date: 2026-07-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b7d8e9f0a1c2"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("picks", sa.Column("closing_point", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("picks", "closing_point")
