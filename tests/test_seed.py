"""The demo rows have to survive someone checking the arithmetic.

A graded record is the product's claim, so seeded figures are only worth
serving if they agree with the functions settle and grade actually use. These
tests recompute every derived number from the stored price and closing
probability rather than trusting the fixture table.
"""

import pytest
from sqlalchemy import select

from fairline.clv import _profit_units
from fairline.db.models import Pick, SteamCandidate
from fairline.seed import DEMO_USER_ID, seed_demo
from fairline.state import american_to_prob
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from fairline.db.models import Base

@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()



@pytest.mark.asyncio
async def test_seed_writes_both_pages_worth_of_rows(session_factory):
    written = await seed_demo(session_factory)

    assert written["picks"] > 0
    assert written["candidates"] > 0

    async with session_factory() as s:
        picks = (await s.execute(select(Pick))).scalars().all()
        cands = (await s.execute(select(SteamCandidate))).scalars().all()

    assert len(picks) == written["picks"]
    assert len(cands) == written["candidates"]


@pytest.mark.asyncio
async def test_every_clv_equals_closing_minus_taken(session_factory):
    """CLV is closing_probability minus the implied probability of the price
    taken. If the seed drifts from that, the record page is lying."""
    await seed_demo(session_factory)

    async with session_factory() as s:
        picks = (await s.execute(select(Pick))).scalars().all()

    assert picks
    for p in picks:
        expected = p.closing_probability - american_to_prob(p.price)
        assert p.clv == pytest.approx(expected, abs=1e-9), (
            f"{p.selection}: stored clv {p.clv} but closing {p.closing_probability} "
            f"minus taken {american_to_prob(p.price)} is {expected}"
        )


@pytest.mark.asyncio
async def test_every_unit_figure_matches_the_grader(session_factory):
    await seed_demo(session_factory)

    async with session_factory() as s:
        picks = (await s.execute(select(Pick))).scalars().all()

    for p in picks:
        assert p.profit_units == pytest.approx(_profit_units(p.price, p.result), abs=1e-9)


@pytest.mark.asyncio
async def test_a_losing_pick_can_still_beat_the_close(session_factory):
    """The reason this product grades on CLV rather than win rate. If the
    fixture set ever loses this property it stops demonstrating the point."""
    await seed_demo(session_factory)

    async with session_factory() as s:
        picks = (await s.execute(select(Pick))).scalars().all()

    losers_beating_close = [p for p in picks if p.result == "loss" and p.clv > 0]
    assert losers_beating_close, "no losing pick with positive CLV, the demo makes no argument"


@pytest.mark.asyncio
async def test_seeding_twice_does_not_double_the_record(session_factory):
    first = await seed_demo(session_factory)
    second = await seed_demo(session_factory)

    assert second == {"picks": 0, "candidates": 0}

    async with session_factory() as s:
        picks = (await s.execute(select(Pick))).scalars().all()
    assert len(picks) == first["picks"]


@pytest.mark.asyncio
async def test_rows_are_attributable_to_the_demo_user(session_factory):
    """So a real deployment can tell sample data from its own and delete it."""
    await seed_demo(session_factory)

    async with session_factory() as s:
        picks = (await s.execute(select(Pick))).scalars().all()

    assert {p.user_id for p in picks} == {DEMO_USER_ID}


@pytest.mark.asyncio
async def test_more_than_one_agent_is_represented(session_factory):
    """The leaderboard compares agents, so one source would make it pointless."""
    await seed_demo(session_factory)

    async with session_factory() as s:
        picks = (await s.execute(select(Pick))).scalars().all()

    assert len({p.source for p in picks}) >= 3
