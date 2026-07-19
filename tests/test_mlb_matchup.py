"""Tests for MLB splits computation: new dimensions, sample-size floor on
vs-pitcher, reuse of matchup.py's sport-agnostic shrinkage math."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from sqlalchemy import select

from fairline.db.models import MlbPlayerGame
from fairline.mlb_matchup import (
    MIN_VS_PITCHER_SAMPLE,
    MLB_PROP_STAT_COLUMNS,
    compute_mlb_prop_splits,
    describe_mlb_splits,
    mlb_matchup_probability,
)


def _game(hits=1, day_night="night", is_home=True, opponent="Boston Red Sox",
          opposing_pitcher="Brayan Bello", team="New York Yankees", season=2025, date=None):
    return MlbPlayerGame(
        season=season,
        game_date=date or datetime(2025, 6, 14, tzinfo=timezone.utc),
        player="Aaron Judge",
        team=team,
        opponent=opponent,
        opposing_pitcher=opposing_pitcher,
        is_home=is_home,
        day_night=day_night,
        at_bats=4,
        hits=hits,
        home_runs=0,
        rbis=0,
        total_bases=hits,
        strikeouts=0,
        walks=0,
    )


class TestComputeMlbPropSplits:
    def test_day_night_home_away_splits_present(self):
        games = [_game(day_night="night", is_home=True), _game(day_night="day", is_home=False, hits=0)]
        splits = compute_mlb_prop_splits(games, "hits", 0.5)
        assert splits["night"] == (1, 1)
        assert splits["day"] == (0, 1)
        assert splits["home"] == (1, 1)
        assert splits["away"] == (0, 1)

    def test_vs_pitcher_below_floor_is_withheld(self):
        games = [_game(hits=1)] * 3  # only 3 PA vs this pitcher, below MIN_VS_PITCHER_SAMPLE
        splits = compute_mlb_prop_splits(games, "hits", 0.5, opposing_pitcher="Brayan Bello")
        assert "vs_pitcher" not in splits

    def test_vs_pitcher_at_or_above_floor_is_included(self):
        games = [_game(hits=1)] * MIN_VS_PITCHER_SAMPLE
        splits = compute_mlb_prop_splits(games, "hits", 0.5, opposing_pitcher="Brayan Bello")
        assert splits["vs_pitcher"] == (MIN_VS_PITCHER_SAMPLE, MIN_VS_PITCHER_SAMPLE)

    def test_vs_pitcher_one_below_floor_is_withheld(self):
        games = [_game(hits=1)] * (MIN_VS_PITCHER_SAMPLE - 1)  # 9 PA, one short of the floor
        splits = compute_mlb_prop_splits(games, "hits", 0.5, opposing_pitcher="Brayan Bello")
        assert "vs_pitcher" not in splits

    def test_park_factor_split_filters_to_matching_bucket(self):
        games = [
            _game(team="Colorado Rockies", opponent="San Diego Padres", is_home=True, hits=1),
            _game(team="Seattle Mariners", opponent="San Diego Padres", is_home=True, hits=0),
        ]
        splits = compute_mlb_prop_splits(games, "hits", 0.5, upcoming_park_bucket="hitter_park")
        assert splits["park_factor"] == (1, 1)  # only the Rockies (hitter-park) game counts

    def test_park_factor_split_matches_away_game_via_opponent_as_host(self):
        # Batter's own team is a neutral park; the game was played on the
        # road at the opponent's park (Colorado, hitter-friendly), so the
        # host resolves through game.opponent, not game.team.
        games = [
            _game(team="Atlanta Braves", opponent="Colorado Rockies", is_home=False, hits=1),
            _game(team="Atlanta Braves", opponent="Seattle Mariners", is_home=False, hits=0),
        ]
        splits = compute_mlb_prop_splits(games, "hits", 0.5, upcoming_park_bucket="hitter_park")
        assert splits["park_factor"] == (1, 1)  # only the road game at Colorado counts

    def test_neutral_upcoming_park_omits_the_split(self):
        games = [_game(team="Atlanta Braves", is_home=True)]
        splits = compute_mlb_prop_splits(games, "hits", 0.5, upcoming_park_bucket=None)
        assert "park_factor" not in splits


def test_mlb_prop_stat_columns_covers_five_batter_markets():
    assert MLB_PROP_STAT_COLUMNS == {
        "batter_hits": "hits",
        "batter_home_runs": "home_runs",
        "batter_rbis": "rbis",
        "batter_total_bases": "total_bases",
        "batter_strikeouts": "strikeouts",
    }


def test_mlb_matchup_probability_bounded_near_market():
    games = [_game(hits=1)] * 10
    prob, splits = mlb_matchup_probability(games, "hits", 0.5, "Over", market_fair=0.55)
    # 0.55 + 0.06 clamp boundary; +1e-9 absorbs the float rounding of that add
    assert 0.49 <= prob <= 0.61 + 1e-9


def test_describe_mlb_splits_lists_only_present_splits():
    splits = {"night": (3, 5), "day": (0, 0)}
    text = describe_mlb_splits(splits, "Over", 0.5)
    assert "night 3-2 over 0.5" in text
    assert "day" not in text  # zero-attempt splits are excluded, same as NFL's describe_splits


def test_player_current_team_resolves_most_recent_game_date():
    from fairline.mlb_matchup import _player_current_team

    older = _game(team="New York Yankees", date=datetime(2025, 5, 1, tzinfo=timezone.utc))
    newer = _game(team="Boston Red Sox", date=datetime(2025, 6, 14, tzinfo=timezone.utc))
    assert _player_current_team([older, newer]) == "Boston Red Sox"
    # Order in the fetched list must not matter.
    assert _player_current_team([newer, older]) == "Boston Red Sox"


@pytest.mark.asyncio
async def test_create_mlb_matchup_candidates_includes_vs_pitcher_when_resolvable(monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base, SteamCandidate
    from fairline.mlb_matchup import create_mlb_matchup_candidates
    from fairline.state import BookmakerOdds, GameSnapshot, MarketOdds, Outcome

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        # Ten prior at-bats against the upcoming probable starter -- enough
        # to clear MIN_VS_PITCHER_SAMPLE and produce a non-degenerate split.
        for i in range(10):
            session.add(_game(
                hits=1, team="New York Yankees", opponent="Boston Red Sox",
                opposing_pitcher="Brayan Bello",
                date=datetime(2025, 5, 1 + i, tzinfo=timezone.utc),
            ))
        await session.commit()

    async def fake_fetch_probable_pitchers(client, date):
        return [{
            "home_team": "New York Yankees", "away_team": "Boston Red Sox",
            "commence_time": datetime(2026, 7, 19, 17, 5, tzinfo=timezone.utc),
            "home_pitcher": None, "away_pitcher": "Brayan Bello",
        }]

    monkeypatch.setattr(
        "fairline.mlb_matchup.fetch_probable_pitchers", fake_fetch_probable_pitchers
    )

    def _o(side, price):
        return Outcome(name=side, price=price, point=0.5, description="Aaron Judge")

    snapshot = GameSnapshot(
        game_id="evt-1", sport="baseball_mlb",
        home_team="New York Yankees", away_team="Boston Red Sox",
        commence_time=datetime(2026, 7, 19, 17, 5, tzinfo=timezone.utc),
        bookmakers=[
            BookmakerOdds(key="pinnacle", title="P", markets=[
                MarketOdds(key="batter_hits", outcomes=[_o("Over", -110), _o("Under", -110)])
            ]),
            BookmakerOdds(key="draftkings", title="DK", markets=[
                MarketOdds(key="batter_hits", outcomes=[_o("Over", 120), _o("Under", -150)])
            ]),
        ],
    )

    created = await create_mlb_matchup_candidates(factory, snapshot, min_edge=0.03)
    assert created == 1

    async with factory() as session:
        cand = (await session.execute(select(SteamCandidate))).scalars().one()
    assert "vs_pitcher" in cand.angles


@pytest.mark.asyncio
async def test_create_mlb_matchup_candidates_omits_vs_pitcher_when_own_team_matches_neither_side(monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base, SteamCandidate
    from fairline.mlb_matchup import create_mlb_matchup_candidates
    from fairline.state import BookmakerOdds, GameSnapshot, MarketOdds, Outcome

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        # Judge's stored team ("New York Yankees") matches neither side below.
        for i in range(10):
            session.add(_game(hits=1, team="New York Yankees", date=datetime(2025, 5, 1 + i, tzinfo=timezone.utc)))
        await session.commit()

    async def fake_fetch_probable_pitchers(client, date):
        return [{
            "home_team": "Miami Marlins", "away_team": "Milwaukee Brewers",
            "commence_time": datetime(2026, 7, 19, 18, 10, tzinfo=timezone.utc),
            "home_pitcher": "Sandy Alcantara", "away_pitcher": "Freddy Peralta",
        }]

    monkeypatch.setattr(
        "fairline.mlb_matchup.fetch_probable_pitchers", fake_fetch_probable_pitchers
    )

    def _o(side, price):
        return Outcome(name=side, price=price, point=0.5, description="Aaron Judge")

    snapshot = GameSnapshot(
        game_id="evt-1", sport="baseball_mlb",
        home_team="Miami Marlins", away_team="Milwaukee Brewers",
        commence_time=datetime(2026, 7, 19, 18, 10, tzinfo=timezone.utc),
        bookmakers=[
            BookmakerOdds(key="pinnacle", title="P", markets=[
                MarketOdds(key="batter_hits", outcomes=[_o("Over", -110), _o("Under", -110)])
            ]),
            BookmakerOdds(key="draftkings", title="DK", markets=[
                MarketOdds(key="batter_hits", outcomes=[_o("Over", 120), _o("Under", -150)])
            ]),
        ],
    )

    created = await create_mlb_matchup_candidates(factory, snapshot, min_edge=0.03)
    assert created == 1

    async with factory() as session:
        cand = (await session.execute(select(SteamCandidate))).scalars().one()
    assert "vs_pitcher" not in (cand.angles or "")


@pytest.mark.asyncio
async def test_create_mlb_matchup_candidates_survives_probable_pitcher_fetch_failure(monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base, SteamCandidate
    from fairline.mlb_matchup import create_mlb_matchup_candidates
    from fairline.state import BookmakerOdds, GameSnapshot, MarketOdds, Outcome

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        # Would otherwise clear MIN_VS_PITCHER_SAMPLE and produce vs_pitcher --
        # the fetch failure below has to suppress it regardless.
        for i in range(10):
            session.add(_game(
                hits=1, team="New York Yankees", opponent="Boston Red Sox",
                opposing_pitcher="Brayan Bello",
                date=datetime(2025, 5, 1 + i, tzinfo=timezone.utc),
            ))
        await session.commit()

    async def fake_fetch_probable_pitchers(client, date):
        raise httpx.HTTPStatusError(
            "500 Server Error", request=httpx.Request("GET", "https://statsapi.mlb.com"),
            response=httpx.Response(500),
        )

    monkeypatch.setattr(
        "fairline.mlb_matchup.fetch_probable_pitchers", fake_fetch_probable_pitchers
    )

    def _o(side, price):
        return Outcome(name=side, price=price, point=0.5, description="Aaron Judge")

    snapshot = GameSnapshot(
        game_id="evt-1", sport="baseball_mlb",
        home_team="New York Yankees", away_team="Boston Red Sox",
        commence_time=datetime(2026, 7, 19, 17, 5, tzinfo=timezone.utc),
        bookmakers=[
            BookmakerOdds(key="pinnacle", title="P", markets=[
                MarketOdds(key="batter_hits", outcomes=[_o("Over", -110), _o("Under", -110)])
            ]),
            BookmakerOdds(key="draftkings", title="DK", markets=[
                MarketOdds(key="batter_hits", outcomes=[_o("Over", 120), _o("Under", -150)])
            ]),
        ],
    )

    created = await create_mlb_matchup_candidates(factory, snapshot, min_edge=0.03)
    assert created == 1

    async with factory() as session:
        cand = (await session.execute(select(SteamCandidate))).scalars().one()
    assert "vs_pitcher" not in (cand.angles or "")


@pytest.mark.asyncio
async def test_create_mlb_matchup_candidates_batches_the_game_fetch_across_players():
    """Two batters on the same slate must each keep their own game history --
    a batched IN-list fetch that mis-groups rows by player would let one
    batter's splits leak into the other's candidate."""
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base, SteamCandidate
    from fairline.mlb_matchup import create_mlb_matchup_candidates
    from fairline.state import BookmakerOdds, GameSnapshot, MarketOdds, Outcome

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        for i in range(10):
            session.add(_game(
                hits=2, team="New York Yankees", opponent="Boston Red Sox",
                date=datetime(2025, 5, 1 + i, tzinfo=timezone.utc),
            ))
            session.add(MlbPlayerGame(
                season=2025, game_date=datetime(2025, 5, 1 + i, tzinfo=timezone.utc),
                player="Juan Soto", team="New York Yankees", opponent="Boston Red Sox",
                opposing_pitcher="Brayan Bello", is_home=True, day_night="night",
                at_bats=4, hits=0, home_runs=0, rbis=0, total_bases=0, strikeouts=0, walks=0,
            ))
        await session.commit()

    async def fake_fetch_probable_pitchers(client, date):
        return []

    import fairline.mlb_matchup as mlb_matchup_module

    original_fetch = mlb_matchup_module.fetch_probable_pitchers
    mlb_matchup_module.fetch_probable_pitchers = fake_fetch_probable_pitchers
    try:
        def _o(side, price, player):
            return Outcome(name=side, price=price, point=0.5, description=player)

        snapshot = GameSnapshot(
            game_id="evt-multi", sport="baseball_mlb",
            home_team="New York Yankees", away_team="Boston Red Sox",
            commence_time=datetime(2026, 7, 19, 17, 5, tzinfo=timezone.utc),
            bookmakers=[
                BookmakerOdds(key="pinnacle", title="P", markets=[
                    MarketOdds(key="batter_hits", outcomes=[
                        _o("Over", -110, "Aaron Judge"), _o("Under", -110, "Aaron Judge"),
                        _o("Over", -110, "Juan Soto"), _o("Under", -110, "Juan Soto"),
                    ])
                ]),
                BookmakerOdds(key="draftkings", title="DK", markets=[
                    MarketOdds(key="batter_hits", outcomes=[
                        _o("Over", 120, "Aaron Judge"), _o("Under", -150, "Aaron Judge"),
                        _o("Under", 120, "Juan Soto"), _o("Over", -150, "Juan Soto"),
                    ])
                ]),
            ],
        )

        batched_game_fetches = 0

        def _count_select(conn, cursor, statement, *args, **kwargs):
            nonlocal batched_game_fetches
            if "mlb_player_games.player IN" in statement:
                batched_game_fetches += 1

        event.listen(engine.sync_engine, "before_cursor_execute", _count_select)
        try:
            created = await create_mlb_matchup_candidates(factory, snapshot, min_edge=0.03)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", _count_select)

        assert batched_game_fetches == 1
        assert created == 2

        async with factory() as session:
            candidates = (await session.execute(select(SteamCandidate))).scalars().all()
        by_player = {c.selection.rsplit(" ", 2)[0]: c for c in candidates}
        assert "last_10 10-0" in by_player["Aaron Judge"].rationale
        assert "last_10 0-10" in by_player["Juan Soto"].rationale
    finally:
        mlb_matchup_module.fetch_probable_pitchers = original_fetch
    await engine.dispose()
