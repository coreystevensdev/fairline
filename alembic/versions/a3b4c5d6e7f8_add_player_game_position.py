"""Add position column to player_games.

Revision ID: a3b4c5d6e7f8
Revises: d4e5f6a7b8c9
Create Date: 2026-07-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a3b4c5d6e7f8"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("player_games", sa.Column("position", sa.String(10), nullable=True))


def downgrade() -> None:
    op.drop_column("player_games", "position")
