"""Tests for NHL splits computation: home/away, back-to-back, and the
sample-size floor on vs-specific-goalie."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from fairline.db.models import NhlPlayerGame
from fairline.nhl_matchup import (
    MIN_VS_GOALIE_SAMPLE,
    NHL_PROP_STAT_COLUMNS,
    compute_nhl_prop_splits,
    describe_nhl_splits,
    nhl_matchup_probability,
)

BASE_DATE = date(2025, 12, 1)


def _game(points=1, is_home=True, rest_days=2, opposing_goalie="Dustin Wolf",
          team="Edmonton Oilers", opponent="Calgary Flames", season=2025, day=0):
    return NhlPlayerGame(
        season=season, game_date=BASE_DATE + timedelta(days=day), player="Connor McDavid",
        team=team, opponent=opponent, opposing_goalie=opposing_goalie,
        is_home=is_home, rest_days=rest_days,
        goals=0, assists=points, points=points, shots_on_goal=points + 2,
    )


class TestComputeNhlPropSplits:
    def test_home_and_away_splits_present(self):
        games = [_game(is_home=True, day=0), _game(is_home=False, day=1, points=0)]
        splits = compute_nhl_prop_splits(games, "points", 0.5)
        assert splits["home"] == (1, 1)
        assert splits["away"] == (0, 1)

    def test_back_to_back_split(self):
        games = [_game(rest_days=1, day=0), _game(rest_days=3, day=1, points=0)]
        splits = compute_nhl_prop_splits(games, "points", 0.5)
        assert splits["back_to_back"] == (1, 1)

    def test_vs_goalie_below_floor_is_withheld(self):
        games = [_game(day=i) for i in range(3)]
        splits = compute_nhl_prop_splits(games, "points", 0.5, opposing_goalie="Dustin Wolf")
        assert "vs_goalie" not in splits

    def test_vs_goalie_one_below_floor_is_withheld(self):
        games = [_game(day=i) for i in range(MIN_VS_GOALIE_SAMPLE - 1)]
        splits = compute_nhl_prop_splits(games, "points", 0.5, opposing_goalie="Dustin Wolf")
        assert "vs_goalie" not in splits

    def test_vs_goalie_at_or_above_floor_is_included(self):
        games = [_game(day=i) for i in range(MIN_VS_GOALIE_SAMPLE)]
        splits = compute_nhl_prop_splits(games, "points", 0.5, opposing_goalie="Dustin Wolf")
        assert splits["vs_goalie"] == (MIN_VS_GOALIE_SAMPLE, MIN_VS_GOALIE_SAMPLE)


def test_nhl_prop_stat_columns_covers_four_skater_markets():
    assert NHL_PROP_STAT_COLUMNS == {
        "player_goals": "goals",
        "player_assists": "assists",
        "player_points": "points",
        "player_shots_on_goal": "shots_on_goal",
    }


def test_nhl_matchup_probability_bounded_near_market():
    games = [_game(day=i) for i in range(10)]
    prob, splits = nhl_matchup_probability(games, "points", 0.5, "Over", market_fair=0.55)
    assert 0.49 <= prob <= 0.61 + 1e-9


def test_describe_nhl_splits_lists_only_present_splits():
    splits = {"home": (3, 5), "away": (0, 0)}
    text = describe_nhl_splits(splits, "Over", 0.5)
    assert "home 3-2 over 0.5" in text
    assert "away" not in text


@pytest.mark.asyncio
async def test_create_nhl_matchup_candidates_batches_the_game_fetch_across_players():
    """Two skaters on the same slate must each keep their own game history in
    the batched IN-list fetch -- a grouping bug would let one skater's splits
    leak into the other's candidate."""
    from sqlalchemy import event, select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base, NhlPlayerGame, SteamCandidate
    from fairline.nhl_matchup import create_nhl_matchup_candidates
    from fairline.state import BookmakerOdds, GameSnapshot, MarketOdds, Outcome

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        for day in range(10):
            session.add(_game(points=3, day=day))  # Connor McDavid: ten games well over 0.5
            session.add(NhlPlayerGame(
                season=2025, game_date=BASE_DATE + timedelta(days=day), player="Auston Matthews",
                team="Edmonton Oilers", opponent="Calgary Flames", opposing_goalie="Dustin Wolf",
                is_home=True, rest_days=2,
                goals=0, assists=0, points=0, shots_on_goal=1,  # never clears 0.5
            ))
        await session.commit()

    def _o(side, price, player):
        return Outcome(name=side, price=price, point=0.5, description=player)

    snapshot = GameSnapshot(
        game_id="evt-multi", sport="icehockey_nhl",
        home_team="Edmonton Oilers", away_team="Calgary Flames", commence_time=BASE_DATE,
        bookmakers=[
            BookmakerOdds(key="pinnacle", title="Pinnacle", markets=[
                MarketOdds(key="player_points", outcomes=[
                    _o("Over", -110, "Connor McDavid"), _o("Under", -110, "Connor McDavid"),
                    _o("Over", -110, "Auston Matthews"), _o("Under", -110, "Auston Matthews"),
                ])
            ]),
            BookmakerOdds(key="fanduel", title="FanDuel", markets=[
                MarketOdds(key="player_points", outcomes=[
                    _o("Over", 150, "Connor McDavid"), _o("Under", -200, "Connor McDavid"),
                    _o("Under", 150, "Auston Matthews"), _o("Over", -200, "Auston Matthews"),
                ])
            ]),
        ],
    )

    batched_game_fetches = 0

    def _count_select(conn, cursor, statement, *args, **kwargs):
        nonlocal batched_game_fetches
        if "nhl_player_games.player IN" in statement:
            batched_game_fetches += 1

    event.listen(engine.sync_engine, "before_cursor_execute", _count_select)
    try:
        created = await create_nhl_matchup_candidates(factory, snapshot, min_edge=0.03)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _count_select)

    assert batched_game_fetches == 1
    assert created == 2

    async with factory() as session:
        candidates = (await session.execute(select(SteamCandidate))).scalars().all()
    by_player = {c.selection.rsplit(" ", 2)[0]: c for c in candidates}
    assert "last_10 10-0" in by_player["Connor McDavid"].rationale
    assert "last_10 0-10" in by_player["Auston Matthews"].rationale
    await engine.dispose()
