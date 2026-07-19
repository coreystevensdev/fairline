"""Tests for NBA splits computation: last-N, season, home/away, back-to-back.
No vs-defender split exists for NBA (no verified per-game defender-matchup
data source), so there is no sample-size-floor test in this module, unlike
the MLB/NHL equivalents."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fairline.db.models import NbaPlayerGame
from fairline.nba_matchup import (
    NBA_PROP_STAT_COLUMNS,
    _player_current_team,
    compute_nba_prop_splits,
    describe_nba_splits,
    nba_matchup_probability,
)

BASE_DATE = datetime(2025, 12, 1, tzinfo=timezone.utc)


def _game(points=25, is_home=True, rest_days=2, season=2025, day=0, position="Forward"):
    return NbaPlayerGame(
        season=season, game_date=BASE_DATE + timedelta(days=day), player="LeBron James",
        team="Los Angeles Lakers", opponent="Boston Celtics",
        is_home=is_home, rest_days=rest_days, position=position,
        points=points, rebounds=8, assists=9, three_pointers_made=3,
    )


class TestComputeNbaPropSplits:
    def test_home_and_away_splits_present(self):
        games = [_game(is_home=True, day=0), _game(is_home=False, day=1, points=15)]
        splits = compute_nba_prop_splits(games, "points", 20.5)
        assert splits["home"] == (1, 1)
        assert splits["away"] == (0, 1)

    def test_back_to_back_split(self):
        games = [_game(rest_days=1, day=0), _game(rest_days=3, day=1, points=15)]
        splits = compute_nba_prop_splits(games, "points", 20.5)
        assert splits["back_to_back"] == (1, 1)

    def test_no_vs_defender_key_exists(self):
        games = [_game(day=0)]
        splits = compute_nba_prop_splits(games, "points", 20.5)
        assert "vs_defender" not in splits
        assert set(splits.keys()) == {"last_5", "last_10", "season", "home", "away", "back_to_back"}


def test_nba_prop_stat_columns_covers_four_markets():
    assert NBA_PROP_STAT_COLUMNS == {
        "player_points": "points",
        "player_rebounds": "rebounds",
        "player_assists": "assists",
        "player_threes": "three_pointers_made",
    }


def test_nba_matchup_probability_bounded_near_market():
    games = [_game(day=i) for i in range(10)]
    prob, splits = nba_matchup_probability(games, "points", 20.5, "Over", market_fair=0.55)
    assert 0.49 <= prob <= 0.61 + 1e-9


def test_describe_nba_splits_lists_only_present_splits():
    splits = {"home": (3, 5), "away": (0, 0)}
    text = describe_nba_splits(splits, "Over", 20.5)
    assert "home 3-2 over 20.5" in text
    assert "away" not in text


class TestPlayerCurrentTeam:
    def test_resolves_most_recent_team_not_fetch_order(self):
        """A traded player's games list can arrive in any order (no ORDER BY
        on the query) -- games[0] would silently return the stale pre-trade
        team here, since the newer row sits second in the list."""
        old_team_game = NbaPlayerGame(
            season=2025, game_date=BASE_DATE, player="LeBron James",
            team="Los Angeles Lakers", opponent="Boston Celtics",
            is_home=True, rest_days=2, position="Forward",
            points=25, rebounds=8, assists=9, three_pointers_made=3,
        )
        new_team_game = NbaPlayerGame(
            season=2025, game_date=BASE_DATE + timedelta(days=30), player="LeBron James",
            team="Dallas Mavericks", opponent="Miami Heat",
            is_home=False, rest_days=2, position="Forward",
            points=20, rebounds=6, assists=7, three_pointers_made=2,
        )
        assert _player_current_team([old_team_game, new_team_game]) == "Dallas Mavericks"
        # Order in the fetched list must not matter.
        assert _player_current_team([new_team_game, old_team_game]) == "Dallas Mavericks"

    def test_resolves_by_season_when_game_dates_tie_across_seasons(self):
        earlier_season = NbaPlayerGame(
            season=2024, game_date=BASE_DATE, player="LeBron James",
            team="Los Angeles Lakers", opponent="Boston Celtics",
            is_home=True, rest_days=2, position="Forward",
            points=25, rebounds=8, assists=9, three_pointers_made=3,
        )
        later_season = NbaPlayerGame(
            season=2025, game_date=BASE_DATE, player="LeBron James",
            team="Dallas Mavericks", opponent="Miami Heat",
            is_home=False, rest_days=2, position="Forward",
            points=20, rebounds=6, assists=7, three_pointers_made=2,
        )
        assert _player_current_team([later_season, earlier_season]) == "Dallas Mavericks"


class TestPositionMatchupSplit:
    def test_position_matchup_included_when_provided(self):
        games = [_game(day=0)]
        splits = compute_nba_prop_splits(games, "points", 20.5, position_matchup=(6, 10))
        assert splits["position_matchup"] == (6, 10)

    def test_position_matchup_absent_when_not_provided(self):
        games = [_game(day=0)]
        splits = compute_nba_prop_splits(games, "points", 20.5)
        assert "position_matchup" not in splits


@pytest.mark.asyncio
async def test_opponent_position_rate_queries_across_players():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base
    from fairline.nba_matchup import _opponent_position_rate

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        session.add_all([
            _game(day=0, points=25),  # LeBron James, vs Boston Celtics, Forward, per _game's defaults
            NbaPlayerGame(
                season=2025, game_date=BASE_DATE + timedelta(days=1), player="Kevin Durant",
                team="Phoenix Suns", opponent="Boston Celtics", is_home=True, rest_days=2,
                position="Forward", points=30, rebounds=7, assists=4, three_pointers_made=2,
            ),
            NbaPlayerGame(
                season=2025, game_date=BASE_DATE + timedelta(days=2), player="Jayson Tatum",
                team="Boston Celtics", opponent="Miami Heat", is_home=True, rest_days=2,
                position="Forward", points=25, rebounds=8, assists=5, three_pointers_made=3,
            ),
        ])
        await session.commit()

    async with factory() as session:
        rate = await _opponent_position_rate(session, "Boston Celtics", "Forward", "points", 20.5)

    # Two Forwards (LeBron, Durant) faced Boston Celtics as their opponent, both scored above 20.5.
    # Tatum's row is irrelevant: his opponent is Miami Heat, not Boston Celtics.
    assert rate == (2, 2)


@pytest.mark.asyncio
async def test_opponent_position_rate_returns_none_with_no_matching_games():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base
    from fairline.nba_matchup import _opponent_position_rate

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        rate = await _opponent_position_rate(session, "Boston Celtics", "Center", "points", 20.5)
    assert rate is None


@pytest.mark.asyncio
async def test_create_nba_matchup_candidates_handles_missing_position_gracefully():
    """A player with no resolved position should still get a pick, just without
    the position_matchup split -- graceful degradation means a candidate still
    gets created, not that the function quietly creates nothing."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base, SteamCandidate
    from fairline.nba_matchup import create_nba_matchup_candidates
    from fairline.state import BookmakerOdds, GameSnapshot, MarketOdds, Outcome

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        session.add(_game(day=0, position=None))
        await session.commit()

    snapshot = GameSnapshot(
        game_id="g1", sport="basketball_nba", home_team="Los Angeles Lakers",
        away_team="Boston Celtics", commence_time=BASE_DATE, bookmakers=[
            BookmakerOdds(key="pinnacle", title="Pinnacle", markets=[
                MarketOdds(key="player_points", outcomes=[
                    Outcome(name="Over", price=-110, point=20.5, description="LeBron James"),
                    Outcome(name="Under", price=-110, point=20.5, description="LeBron James"),
                ])
            ]),
            BookmakerOdds(key="fanduel", title="FanDuel", markets=[
                MarketOdds(key="player_points", outcomes=[
                    Outcome(name="Over", price=+150, point=20.5, description="LeBron James"),
                    Outcome(name="Under", price=-200, point=20.5, description="LeBron James"),
                ])
            ]),
        ],
    )

    # Should not raise even though the only stored game has position=None,
    # and the fanduel Over price (+150, implied 0.40) lags the splits-adjusted
    # ~0.545 probability by enough to clear min_edge, so a pick must land.
    created = await create_nba_matchup_candidates(factory, snapshot, min_edge=0.03)
    assert created > 0

    async with factory() as session:
        rows = (await session.execute(select(SteamCandidate))).scalars().all()
    assert len(rows) == created
    assert all("position_matchup" not in (row.angles or "") for row in rows)


@pytest.mark.asyncio
async def test_create_nba_matchup_candidates_omits_split_when_own_team_matches_neither_side():
    """If the player's resolved current team is neither the snapshot's home nor
    away team (a name mismatch, or a prop feed lagging a trade), the opponent
    can't be derived safely -- the split must be omitted, not guessed."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base, SteamCandidate
    from fairline.nba_matchup import create_nba_matchup_candidates
    from fairline.state import BookmakerOdds, GameSnapshot, MarketOdds, Outcome

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        # Player's stored team ("Los Angeles Lakers") matches neither side below.
        session.add(_game(day=0, position="Forward"))
        await session.commit()

    snapshot = GameSnapshot(
        game_id="g1", sport="basketball_nba", home_team="Miami Heat",
        away_team="Denver Nuggets", commence_time=BASE_DATE, bookmakers=[
            BookmakerOdds(key="pinnacle", title="Pinnacle", markets=[
                MarketOdds(key="player_points", outcomes=[
                    Outcome(name="Over", price=-110, point=20.5, description="LeBron James"),
                    Outcome(name="Under", price=-110, point=20.5, description="LeBron James"),
                ])
            ]),
            BookmakerOdds(key="fanduel", title="FanDuel", markets=[
                MarketOdds(key="player_points", outcomes=[
                    Outcome(name="Over", price=+150, point=20.5, description="LeBron James"),
                    Outcome(name="Under", price=-200, point=20.5, description="LeBron James"),
                ])
            ]),
        ],
    )

    created = await create_nba_matchup_candidates(factory, snapshot, min_edge=0.03)
    assert created > 0

    async with factory() as session:
        rows = (await session.execute(select(SteamCandidate))).scalars().all()
    assert len(rows) == created
    assert all("position_matchup" not in (row.angles or "") for row in rows)


@pytest.mark.asyncio
async def test_create_nba_matchup_candidates_batches_the_game_fetch_across_players():
    """Two players on the same slate must each keep their own game history in
    the batched IN-list fetch -- a grouping bug would let one player's splits
    leak into the other's candidate."""
    from sqlalchemy import event, select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base, SteamCandidate
    from fairline.nba_matchup import create_nba_matchup_candidates
    from fairline.state import BookmakerOdds, GameSnapshot, MarketOdds, Outcome

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        for day in range(10):
            session.add(_game(points=30, day=day))  # LeBron James: ten games well over 20.5
            session.add(NbaPlayerGame(
                season=2025, game_date=BASE_DATE + timedelta(days=day), player="Jayson Tatum",
                team="Los Angeles Lakers", opponent="Boston Celtics",
                is_home=True, rest_days=2, position="Forward",
                points=5, rebounds=8, assists=9, three_pointers_made=3,  # well under 20.5
            ))
        await session.commit()

    def _o(side, price, player):
        return Outcome(name=side, price=price, point=20.5, description=player)

    snapshot = GameSnapshot(
        game_id="evt-multi", sport="basketball_nba",
        home_team="Los Angeles Lakers", away_team="Boston Celtics", commence_time=BASE_DATE,
        bookmakers=[
            BookmakerOdds(key="pinnacle", title="Pinnacle", markets=[
                MarketOdds(key="player_points", outcomes=[
                    _o("Over", -110, "LeBron James"), _o("Under", -110, "LeBron James"),
                    _o("Over", -110, "Jayson Tatum"), _o("Under", -110, "Jayson Tatum"),
                ])
            ]),
            BookmakerOdds(key="fanduel", title="FanDuel", markets=[
                MarketOdds(key="player_points", outcomes=[
                    _o("Over", 150, "LeBron James"), _o("Under", -200, "LeBron James"),
                    _o("Under", 150, "Jayson Tatum"), _o("Over", -200, "Jayson Tatum"),
                ])
            ]),
        ],
    )

    batched_game_fetches = 0

    def _count_select(conn, cursor, statement, *args, **kwargs):
        nonlocal batched_game_fetches
        if "nba_player_games.player IN" in statement:
            batched_game_fetches += 1

    event.listen(engine.sync_engine, "before_cursor_execute", _count_select)
    try:
        created = await create_nba_matchup_candidates(factory, snapshot, min_edge=0.03)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _count_select)

    assert batched_game_fetches == 1
    assert created == 2

    async with factory() as session:
        candidates = (await session.execute(select(SteamCandidate))).scalars().all()
    by_player = {c.selection.rsplit(" ", 2)[0]: c for c in candidates}
    assert "last_10 10-0" in by_player["LeBron James"].rationale
    assert "last_10 0-10" in by_player["Jayson Tatum"].rationale
    await engine.dispose()
