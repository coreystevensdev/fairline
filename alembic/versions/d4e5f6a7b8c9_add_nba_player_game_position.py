"""Add position column to nba_player_games.

Revision ID: d4e5f6a7b8c9
Revises: e5f6a7b8c9d0
Create Date: 2026-07-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("nba_player_games", sa.Column("position", sa.String(10), nullable=True))


def downgrade() -> None:
    op.drop_column("nba_player_games", "position")
