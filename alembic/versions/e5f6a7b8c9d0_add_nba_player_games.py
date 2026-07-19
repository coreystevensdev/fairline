"""Add nba_player_games: per-game player stat lines for NBA prop analysis.

Revision ID: e5f6a7b8c9d0
Revises: d2e3f4a5b6c7
Create Date: 2026-07-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "d2e3f4a5b6c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "nba_player_games",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("game_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("player", sa.String(100), nullable=False),
        sa.Column("team", sa.String(100), nullable=False),
        sa.Column("opponent", sa.String(100), nullable=False),
        sa.Column("is_home", sa.Boolean(), nullable=False),
        sa.Column("rest_days", sa.Integer(), nullable=True),
        sa.Column("points", sa.Integer(), nullable=True),
        sa.Column("rebounds", sa.Integer(), nullable=True),
        sa.Column("assists", sa.Integer(), nullable=True),
        sa.Column("three_pointers_made", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_nba_player_games_player", "nba_player_games", ["player", "season"]
    )


def downgrade() -> None:
    op.drop_index("ix_nba_player_games_player", table_name="nba_player_games")
    op.drop_table("nba_player_games")
