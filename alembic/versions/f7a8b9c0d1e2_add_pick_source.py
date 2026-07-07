"""Add picks.source: the producing agent, for the per-agent leaderboard.

Revision ID: f7a8b9c0d1e2
Revises: e6a7b8c9d0f1
Create Date: 2026-07-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "f7a8b9c0d1e2"
down_revision: Union[str, None] = "e6a7b8c9d0f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "picks",
        sa.Column("source", sa.String(20), nullable=False, server_default="model"),
    )


def downgrade() -> None:
    op.drop_column("picks", "source")
