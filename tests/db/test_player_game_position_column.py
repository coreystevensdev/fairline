"""Round-trip test for PlayerGame's new position column."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from fairline.db.models import Base, PlayerGame


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def test_position_column_round_trips(session_factory):
    row = PlayerGame(
        sport="americanfootball_nfl", season=2025, week=10,
        game_date=datetime(2025, 12, 1, tzinfo=timezone.utc),
        player="Patrick Mahomes", team="Kansas City Chiefs", opponent="Las Vegas Raiders",
        position="QB", passing_yards=300.0,
    )
    async with session_factory() as session:
        session.add(row)
        await session.commit()

    async with session_factory() as session:
        saved = (await session.execute(select(PlayerGame))).scalars().first()
    assert saved.position == "QB"


async def test_position_defaults_to_null(session_factory):
    row = PlayerGame(
        sport="americanfootball_nfl", season=2025, week=10,
        player="Old Row", team="Kansas City Chiefs", opponent="Las Vegas Raiders",
    )
    async with session_factory() as session:
        session.add(row)
        await session.commit()

    async with session_factory() as session:
        saved = (await session.execute(select(PlayerGame))).scalars().first()
    assert saved.position is None
