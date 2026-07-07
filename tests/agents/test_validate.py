"""Tests for validate_agent: DB persistence and graceful no-op paths.

Uses an in-memory SQLite database via aiosqlite so no Postgres is needed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from steambot.agents.validate import validate_agent
from steambot.db.models import Base, Pick
from steambot.state import ApprovedPick, PickCandidate


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def _make_approved(pick_id: str = "pick-1") -> ApprovedPick:
    candidate = PickCandidate(
        pick_id=pick_id,
        game_id="game-1",
        home_team="Kansas City Chiefs",
        away_team="Las Vegas Raiders",
        commence_time=datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc),
        market="spreads",
        selection="Kansas City Chiefs -3.5",
        best_book="draftkings",
        best_price=-108,
        sharp_probability=0.545,
        blended_probability=0.545,
        implied_probability=0.519,
        edge_pct=0.026,
        ev_pct=0.031,
        confidence="medium",
        rationale="Sharp line moved against public action.",
    )
    return ApprovedPick(
        pick=candidate,
        approved_at=datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc),
        user_id="user-1",
    )


def _state(approved: list[ApprovedPick]) -> dict:
    return {
        "sport": "americanfootball_nfl",
        "target_date": "2026-01-15",
        "user_id": "user-1",
        "games": [],
        "fair_lines": [],
        "candidates": [],
        "approved_picks": approved,
        "bet_slips": [],
        "run_id": "run-1",
        "error": None,
    }


async def test_validate_agent_writes_pick_to_db(session_factory):
    await validate_agent(_state([_make_approved()]), session_factory=session_factory)

    async with session_factory() as session:
        rows = (await session.execute(select(Pick))).scalars().all()

    assert len(rows) == 1
    p = rows[0]
    assert p.id == "pick-1"
    assert p.selection == "Kansas City Chiefs -3.5"
    assert p.edge_pct == pytest.approx(0.026)
    assert p.confidence == "medium"
    assert p.run_id == "run-1"


async def test_validate_agent_no_approved_skips_db(session_factory):
    result = await validate_agent(_state([]), session_factory=session_factory)

    assert result == {}
    async with session_factory() as session:
        rows = (await session.execute(select(Pick))).scalars().all()
    assert len(rows) == 0


async def test_validate_agent_rerun_is_idempotent(session_factory):
    state = _state([_make_approved()])
    await validate_agent(state, session_factory=session_factory)
    await validate_agent(state, session_factory=session_factory)

    async with session_factory() as session:
        rows = (await session.execute(select(Pick))).scalars().all()
    assert len(rows) == 1


async def test_validate_agent_no_factory_skips_gracefully():
    approved = _make_approved()
    result = await validate_agent(_state([approved]), session_factory=None)
    assert result == {}
