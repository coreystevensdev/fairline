"""Initial schema: users and picks tables.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-07-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("stripe_customer_id", sa.String(255), nullable=True),
        sa.Column("is_pro", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "picks",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("game_id", sa.String(255), nullable=False),
        sa.Column("home_team", sa.String(100), nullable=True),
        sa.Column("away_team", sa.String(100), nullable=True),
        sa.Column("commence_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("market", sa.String(50), nullable=True),
        sa.Column("selection", sa.String(255), nullable=True),
        sa.Column("book", sa.String(100), nullable=True),
        sa.Column("price", sa.Integer(), nullable=True),
        sa.Column("sharp_probability", sa.Float(), nullable=True),
        sa.Column("blended_probability", sa.Float(), nullable=True),
        sa.Column("edge_pct", sa.Float(), nullable=True),
        sa.Column("ev_pct", sa.Float(), nullable=True),
        sa.Column("confidence", sa.String(20), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closing_price", sa.Integer(), nullable=True),
        sa.Column("closing_probability", sa.Float(), nullable=True),
        sa.Column("clv", sa.Float(), nullable=True),
        sa.Column("result", sa.String(10), nullable=True),
        sa.Column("profit_units", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_picks_user_id", "picks", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_picks_user_id", table_name="picks")
    op.drop_table("picks")
    op.drop_table("users")
