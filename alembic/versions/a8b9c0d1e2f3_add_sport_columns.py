"""Add sport to picks and line_snapshots for multi-league support.

Revision ID: a8b9c0d1e2f3
Revises: f7a8b9c0d1e2
Create Date: 2026-07-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a8b9c0d1e2f3"
down_revision: Union[str, None] = "f7a8b9c0d1e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("picks", "line_snapshots"):
        op.add_column(
            table,
            sa.Column(
                "sport",
                sa.String(50),
                nullable=False,
                server_default="americanfootball_nfl",
            ),
        )


def downgrade() -> None:
    op.drop_column("line_snapshots", "sport")
    op.drop_column("picks", "sport")
