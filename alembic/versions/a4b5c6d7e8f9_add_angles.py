"""Add angles to picks and steam_candidates for per-angle CLV attribution.

Revision ID: a4b5c6d7e8f9
Revises: f3a4b5c6d7e8
Create Date: 2026-07-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a4b5c6d7e8f9"
down_revision: Union[str, None] = "f3a4b5c6d7e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("picks", sa.Column("angles", sa.Text(), nullable=True))
    op.add_column("steam_candidates", sa.Column("angles", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("steam_candidates", "angles")
    op.drop_column("picks", "angles")
