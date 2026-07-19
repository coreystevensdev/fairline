"""Model-shape test for NbaPlayerGame: confirms columns exist and round-trips through sqlite."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from fairline.db.models import Base, NbaPlayerGame


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def test_nba_player_game_round_trips(session_factory):
    row = NbaPlayerGame(
        season=2025,
        game_date=datetime(2025, 12, 1, tzinfo=timezone.utc),
        player="LeBron James",
        team="Los Angeles Lakers",
        opponent="Boston Celtics",
        is_home=True,
        rest_days=2,
        points=28,
        rebounds=8,
        assists=9,
        three_pointers_made=3,
    )
    async with session_factory() as session:
        session.add(row)
        await session.commit()

    async with session_factory() as session:
        saved = (await session.execute(select(NbaPlayerGame))).scalars().first()
    assert saved.player == "LeBron James"
    assert saved.is_home is True
    assert saved.points == 28
    assert saved.rest_days == 2
