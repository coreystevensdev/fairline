"""Tests for the ratings-based simulation model and its agent node."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from fairline.db.models import Base, GameResult
from fairline.sim import (
    HFA_POINTS,
    SIGMA_MARGIN,
    build_ratings,
    cover_probability,
    parse_nflverse_games,
    season_of,
    sim_agent,
    win_probability,
)
from fairline.state import BookmakerOdds, GameSnapshot, MarketOdds, Outcome, SimLine

NOW = datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc)


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


class TestMarginModel:
    def test_equal_teams_at_home_win_about_56_percent(self):
        # design doc pin: expected margin = HFA, sigma 13.5
        p = win_probability(HFA_POINTS)
        assert p == pytest.approx(0.5589, abs=0.002)

    def test_zero_margin_is_a_coin_flip(self):
        assert win_probability(0.0) == pytest.approx(0.5)

    def test_cover_probability_laying_points(self):
        # 2-point better team laying 3.5: P(margin > 3.5) = phi((2 - 3.5) / sigma)
        p = cover_probability(expected_margin=2.0, team_point=-3.5)
        assert p == pytest.approx(0.4558, abs=0.002)

    def test_cover_probability_getting_points(self):
        p = cover_probability(expected_margin=-2.0, team_point=3.5)
        assert p == pytest.approx(0.5442, abs=0.002)


class TestRatings:
    def test_winner_gains_and_loser_drops(self):
        games = [
            {"season": 2025, "home_team": "A", "away_team": "B", "home_score": 30, "away_score": 10},
        ]
        ratings = build_ratings(games)
        assert ratings["A"] > 0 > ratings["B"]

    def test_repeated_blowouts_converge_toward_the_margin(self):
        games = [
            {"season": 2025, "home_team": "A", "away_team": "B", "home_score": 24, "away_score": 10}
            for _ in range(60)
        ]
        ratings = build_ratings(games)
        # A beats B by 14 at home; rating gap should approach 14 - HFA = 12
        assert ratings["A"] - ratings["B"] == pytest.approx(12.0, abs=2.0)

    def test_season_boundary_regresses_toward_zero(self):
        one_season = build_ratings(
            [{"season": 2024, "home_team": "A", "away_team": "B", "home_score": 30, "away_score": 10}]
        )
        crossed = build_ratings(
            [
                {"season": 2024, "home_team": "A", "away_team": "B", "home_score": 30, "away_score": 10},
                {"season": 2025, "home_team": "C", "away_team": "D", "home_score": 20, "away_score": 17},
            ]
        )
        assert abs(crossed["A"]) < abs(one_season["A"])


def test_season_of_maps_january_to_prior_season():
    assert season_of(datetime(2026, 1, 15, tzinfo=timezone.utc)) == 2025
    assert season_of(datetime(2025, 10, 5, tzinfo=timezone.utc)) == 2025


def test_parse_nflverse_games_maps_codes_and_lines():
    csv_text = (
        "game_id,season,week,gameday,home_team,away_team,home_score,away_score,spread_line,total_line\n"
        "2025_10_LV_KC,2025,10,2025-11-09,KC,LV,27,20,3.5,44.5\n"
        "2025_11_KC_DEN,2025,11,2025-11-16,DEN,KC,,,2.5,41.0\n"
    )
    sim_games, results = parse_nflverse_games(csv_text)

    assert len(sim_games) == 1  # unscored future game skipped
    g = sim_games[0]
    assert g["home_team"] == "Kansas City Chiefs"
    assert g["away_team"] == "Las Vegas Raiders"

    assert len(results) == 1
    r = results[0]
    assert r.game_id == "2025_10_LV_KC"
    # spread_line 3.5 = home favored by 3.5 -> home handicap -3.5
    assert r.closing_spread_home == -3.5
    assert r.closing_total == 44.5


async def test_sim_agent_writes_h2h_and_spread_lines(session_factory):
    async with session_factory() as session:
        for i in range(1, 9):
            session.add(
                GameResult(
                    game_id=f"h{i}",
                    sport="americanfootball_nfl",
                    home_team="Kansas City Chiefs",
                    away_team="Denver Broncos",
                    commence_time=NOW - timedelta(days=7 * i),
                    home_score=30,
                    away_score=13,
                )
            )
        await session.commit()

    game = GameSnapshot(
        game_id="up-1",
        sport="americanfootball_nfl",
        home_team="Kansas City Chiefs",
        away_team="Denver Broncos",
        commence_time=NOW + timedelta(days=2),
        bookmakers=[
            BookmakerOdds(
                key="pinnacle",
                title="Pinnacle",
                markets=[
                    MarketOdds(
                        key="spreads",
                        outcomes=[
                            Outcome(name="Kansas City Chiefs", price=-110, point=-7.5),
                            Outcome(name="Denver Broncos", price=-110, point=7.5),
                        ],
                    )
                ],
            )
        ],
    )
    state = {"sport": "americanfootball_nfl", "games": [game], "sim_lines": []}

    out = await sim_agent(state, session_factory=session_factory)

    lines = out["sim_lines"]
    h2h = [sl for sl in lines if sl.market == "h2h" and sl.selection == "Kansas City Chiefs"]
    spreads = [sl for sl in lines if sl.market == "spreads" and sl.selection.startswith("Kansas City Chiefs")]
    assert len(h2h) == 1 and h2h[0].probability > 0.6
    assert len(spreads) == 1 and 0.0 < spreads[0].probability < 1.0


async def test_sim_agent_defers_to_caller_supplied_lines(session_factory):
    async with session_factory() as session:
        session.add(
            GameResult(
                game_id="h1",
                sport="americanfootball_nfl",
                home_team="Kansas City Chiefs",
                away_team="Denver Broncos",
                commence_time=NOW - timedelta(days=7),
                home_score=30,
                away_score=13,
            )
        )
        await session.commit()

    caller_line = SimLine(
        home_team="Kansas City Chiefs",
        away_team="Denver Broncos",
        market="h2h",
        selection="Kansas City Chiefs",
        probability=0.99,
    )
    game = GameSnapshot(
        game_id="up-1",
        sport="americanfootball_nfl",
        home_team="Kansas City Chiefs",
        away_team="Denver Broncos",
        commence_time=NOW + timedelta(days=2),
        bookmakers=[],
    )
    state = {"sport": "americanfootball_nfl", "games": [game], "sim_lines": [caller_line]}

    out = await sim_agent(state, session_factory=session_factory)

    h2h = [sl for sl in out["sim_lines"] if sl.market == "h2h"]
    assert len(h2h) == 1
    assert h2h[0].probability == 0.99


async def test_sim_agent_without_db_passes_caller_lines_through():
    line = SimLine(
        home_team="A", away_team="B", market="h2h", selection="A", probability=0.6
    )
    out = await sim_agent({"games": [], "sim_lines": [line]}, session_factory=None)
    assert out == {"sim_lines": [line]}


class TestTotalsModel:
    def _games(self):
        # A: high-scoring both ways; B: low-scoring. League avg pts = (30+20+10+14) / 4 = 18.5
        return [
            {"season": 2025, "home_team": "A", "away_team": "B", "home_score": 30, "away_score": 10},
            {"season": 2025, "home_team": "B", "away_team": "A", "home_score": 14, "away_score": 20},
        ]

    def test_scoring_rates_center_on_league_average(self):
        from fairline.sim import build_scoring_rates

        rates, league_avg = build_scoring_rates(self._games())

        assert league_avg == pytest.approx(18.5)
        # A scored 30 and 20 -> off dev +6.5; allowed 10 and 14 -> def dev -6.5
        assert rates["A"]["off"] == pytest.approx(6.5)
        assert rates["A"]["def"] == pytest.approx(-6.5)
        assert rates["B"]["off"] == pytest.approx(-6.5)

    def test_expected_total_sums_deviations(self):
        from fairline.sim import build_scoring_rates, expected_total

        rates, league_avg = build_scoring_rates(self._games())
        # 2*18.5 + offA 6.5 + offB -6.5 + defA -6.5 + defB 6.5 = 37.0
        assert expected_total(rates, league_avg, "A", "B") == pytest.approx(37.0)

    def test_unknown_team_uses_league_average(self):
        from fairline.sim import build_scoring_rates, expected_total

        rates, league_avg = build_scoring_rates(self._games())
        assert expected_total(rates, league_avg, "X", "Y") == pytest.approx(37.0)

    def test_over_probability_pin(self):
        from fairline.sim import over_probability

        # expected 47, line 44.5, sigma 10 -> phi(0.25) = 0.5987
        assert over_probability(47.0, 44.5) == pytest.approx(0.5987, abs=0.002)

    def test_rates_use_latest_season_only(self):
        from fairline.sim import build_scoring_rates

        games = self._games() + [
            {"season": 2024, "home_team": "A", "away_team": "B", "home_score": 60, "away_score": 60}
        ]
        rates, league_avg = build_scoring_rates(games)
        assert league_avg == pytest.approx(18.5)


async def test_sim_agent_emits_totals_line(session_factory):
    async with session_factory() as session:
        for i in range(1, 5):
            session.add(
                GameResult(
                    game_id=f"t{i}",
                    sport="americanfootball_nfl",
                    home_team="Kansas City Chiefs",
                    away_team="Denver Broncos",
                    commence_time=NOW - timedelta(days=7 * i),
                    home_score=30,
                    away_score=20,
                )
            )
        await session.commit()

    game = GameSnapshot(
        game_id="up-1",
        sport="americanfootball_nfl",
        home_team="Kansas City Chiefs",
        away_team="Denver Broncos",
        commence_time=NOW + timedelta(days=2),
        bookmakers=[
            BookmakerOdds(
                key="pinnacle",
                title="Pinnacle",
                markets=[
                    MarketOdds(
                        key="totals",
                        outcomes=[
                            Outcome(name="Over", price=-110, point=44.5),
                            Outcome(name="Under", price=-110, point=44.5),
                        ],
                    )
                ],
            )
        ],
    )
    state = {"sport": "americanfootball_nfl", "games": [game], "sim_lines": []}

    out = await sim_agent(state, session_factory=session_factory)

    totals = [sl for sl in out["sim_lines"] if sl.market == "totals"]
    assert len(totals) == 1
    assert totals[0].selection == "Over 44.5"
    # every game totals 50 against a 44.5 line; the model should lean over
    assert totals[0].probability > 0.6


class TestSportModels:
    def test_nba_uses_tighter_margin_sigma(self):
        from fairline.sim import SPORT_MODELS, win_probability_for

        nba = SPORT_MODELS["basketball_nba"]
        # equal teams at home: phi(2.5 / 11.5) = 0.586
        p = win_probability_for("basketball_nba", nba["hfa"])
        assert p == pytest.approx(0.5861, abs=0.002)

    def test_poisson_even_matchup_is_a_coin_flip(self):
        # home edge enters via the lambdas, not this function: equal rates = 0.5
        from fairline.sim import poisson_win_probability

        assert poisson_win_probability(3.0, 3.0) == pytest.approx(0.5, abs=1e-9)

    def test_poisson_stronger_team_wins_more(self):
        from fairline.sim import poisson_win_probability

        assert poisson_win_probability(4.0, 2.5) > 0.65

    def test_poisson_total_over_probability(self):
        from fairline.sim import poisson_over_probability

        # lambda 6.0 vs 5.5 line: P(total >= 6) with Poisson(6) = 1 - CDF(5) = 0.5543
        assert poisson_over_probability(6.0, 5.5) == pytest.approx(0.5543, abs=0.002)

    def test_poisson_cover_probability_puck_line(self):
        from fairline.sim import poisson_cover_probability

        # home -1.5 needs a 2+ goal win; favorites do that well under half the time
        p = poisson_cover_probability(3.5, 2.5, -1.5)
        assert 0.25 < p < 0.55


async def test_sim_agent_covers_nba_with_normal_family(session_factory):
    async with session_factory() as session:
        for i in range(1, 7):
            session.add(
                GameResult(
                    game_id=f"nba{i}",
                    sport="basketball_nba",
                    home_team="Boston Celtics",
                    away_team="Washington Wizards",
                    commence_time=NOW - timedelta(days=3 * i),
                    home_score=120,
                    away_score=100,
                )
            )
        await session.commit()

    game = GameSnapshot(
        game_id="nba-up",
        sport="basketball_nba",
        home_team="Boston Celtics",
        away_team="Washington Wizards",
        commence_time=NOW + timedelta(days=1),
        bookmakers=[],
    )
    state = {"sport": "basketball_nba", "games": [game], "sim_lines": []}

    out = await sim_agent(state, session_factory=session_factory)

    h2h = [sl for sl in out["sim_lines"] if sl.market == "h2h"]
    assert len(h2h) == 1
    assert h2h[0].probability > 0.7


async def test_sim_agent_covers_nhl_with_poisson_family(session_factory):
    async with session_factory() as session:
        for i in range(1, 7):
            session.add(
                GameResult(
                    game_id=f"nhl{i}",
                    sport="icehockey_nhl",
                    home_team="Boston Bruins",
                    away_team="San Jose Sharks",
                    commence_time=NOW - timedelta(days=3 * i),
                    home_score=4,
                    away_score=2,
                )
            )
        await session.commit()

    game = GameSnapshot(
        game_id="nhl-up",
        sport="icehockey_nhl",
        home_team="Boston Bruins",
        away_team="San Jose Sharks",
        commence_time=NOW + timedelta(days=1),
        bookmakers=[],
    )
    state = {"sport": "icehockey_nhl", "games": [game], "sim_lines": []}

    out = await sim_agent(state, session_factory=session_factory)

    h2h = [sl for sl in out["sim_lines"] if sl.market == "h2h"]
    assert len(h2h) == 1
    assert h2h[0].probability > 0.6


async def test_sim_agent_applies_wind_to_totals(session_factory):
    async with session_factory() as session:
        for i in range(1, 5):
            session.add(
                GameResult(
                    game_id=f"w{i}",
                    sport="americanfootball_nfl",
                    home_team="Kansas City Chiefs",
                    away_team="Denver Broncos",
                    commence_time=NOW - timedelta(days=7 * i),
                    home_score=30,
                    away_score=20,
                )
            )
        await session.commit()

    game = GameSnapshot(
        game_id="windy-1",
        sport="americanfootball_nfl",
        home_team="Kansas City Chiefs",
        away_team="Denver Broncos",
        commence_time=NOW + timedelta(days=2),
        bookmakers=[
            BookmakerOdds(
                key="pinnacle",
                title="Pinnacle",
                markets=[
                    MarketOdds(
                        key="totals",
                        outcomes=[
                            Outcome(name="Over", price=-110, point=44.5),
                            Outcome(name="Under", price=-110, point=44.5),
                        ],
                    )
                ],
            )
        ],
    )
    base_state = {"sport": "americanfootball_nfl", "games": [game], "sim_lines": []}
    calm = await sim_agent(dict(base_state), session_factory=session_factory)
    windy = await sim_agent(
        {**base_state, "game_weather": {"windy-1": {"wind_mph": 25.0}}},
        session_factory=session_factory,
    )

    calm_over = next(sl for sl in calm["sim_lines"] if sl.market == "totals").probability
    windy_over = next(sl for sl in windy["sim_lines"] if sl.market == "totals").probability
    assert windy_over < calm_over


class TestRestAdjustment:
    def test_nba_back_to_back_costs_points(self):
        from fairline.sim import rest_margin_adjustment

        # home on a b2b, away rested: home margin expectation drops
        adj = rest_margin_adjustment(
            "basketball_nba", {"b2b": True, "rest_days": 1}, {"b2b": False, "rest_days": 3}
        )
        assert adj == pytest.approx(-2.0)

    def test_nfl_short_week_and_bye_bump(self):
        from fairline.sim import rest_margin_adjustment

        short = rest_margin_adjustment(
            "americanfootball_nfl", {"rest_days": 4}, {"rest_days": 7}
        )
        rested = rest_margin_adjustment(
            "americanfootball_nfl", {"rest_days": 14}, {"rest_days": 7}
        )
        assert short == pytest.approx(-1.0)
        assert rested == pytest.approx(1.0)

    def test_missing_context_is_neutral(self):
        from fairline.sim import rest_margin_adjustment

        assert rest_margin_adjustment("basketball_nba", {}, {}) == 0.0


async def test_sim_agent_reads_rest_from_team_trends(session_factory):
    async with session_factory() as session:
        for i in range(1, 5):
            session.add(
                GameResult(
                    game_id=f"r{i}",
                    sport="basketball_nba",
                    home_team="Boston Celtics",
                    away_team="Washington Wizards",
                    commence_time=NOW - timedelta(days=3 * i),
                    home_score=110,
                    away_score=110,
                )
            )
        await session.commit()

    game = GameSnapshot(
        game_id="nba-rest",
        sport="basketball_nba",
        home_team="Boston Celtics",
        away_team="Washington Wizards",
        commence_time=NOW + timedelta(days=1),
        bookmakers=[],
    )
    base = {"sport": "basketball_nba", "games": [game], "sim_lines": []}
    neutral = await sim_agent(dict(base), session_factory=session_factory)
    home_b2b = await sim_agent(
        {**base, "team_trends": {"Boston Celtics": {"b2b": True, "rest_days": 1},
                                 "Washington Wizards": {"b2b": False, "rest_days": 3}}},
        session_factory=session_factory,
    )

    p_neutral = next(sl for sl in neutral["sim_lines"] if sl.market == "h2h").probability
    p_b2b = next(sl for sl in home_b2b["sim_lines"] if sl.market == "h2h").probability
    assert p_b2b < p_neutral


async def test_sim_agent_applies_injury_adjustment(session_factory):
    async with session_factory() as session:
        for i in range(1, 5):
            session.add(
                GameResult(
                    game_id=f"inj{i}",
                    sport="americanfootball_nfl",
                    home_team="Kansas City Chiefs",
                    away_team="Denver Broncos",
                    commence_time=NOW - timedelta(days=7 * i),
                    home_score=27,
                    away_score=20,
                )
            )
        await session.commit()

    game = GameSnapshot(
        game_id="inj-up",
        sport="americanfootball_nfl",
        home_team="Kansas City Chiefs",
        away_team="Denver Broncos",
        commence_time=NOW + timedelta(days=2),
        bookmakers=[],
    )
    base = {"sport": "americanfootball_nfl", "games": [game], "sim_lines": []}
    healthy = await sim_agent(dict(base), session_factory=session_factory)
    qb_out = await sim_agent(
        {**base, "team_injuries": {"Kansas City Chiefs": {"adjustment": -5.5, "notes": ["QB Out"]}}},
        session_factory=session_factory,
    )

    p_healthy = next(sl for sl in healthy["sim_lines"] if sl.market == "h2h").probability
    p_qb_out = next(sl for sl in qb_out["sim_lines"] if sl.market == "h2h").probability
    assert p_qb_out < p_healthy - 0.1
