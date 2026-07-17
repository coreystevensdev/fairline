"""Tests for stats_agent: team resolution, graceful degradation on fetch failure."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from fairline.agents.stats import stats_agent
from fairline.state import GameSnapshot

_BASE = "https://api.balldontlie.io"


def _game(home="Kansas City Chiefs", away="Las Vegas Raiders", sport="americanfootball_nfl") -> GameSnapshot:
    return GameSnapshot(
        game_id="g1",
        sport=sport,
        home_team=home,
        away_team=away,
        commence_time=datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc),
        bookmakers=[],
    )


_TEAMS = {
    "data": [
        {"id": 1, "full_name": "Kansas City Chiefs"},
        {"id": 2, "full_name": "Las Vegas Raiders"},
    ]
}


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("BALLDONTLIE_API_KEY", "test-key")


@pytest.mark.asyncio
@respx.mock
async def test_stats_agent_fetches_team_and_player_stats_for_nfl():
    respx.get(f"{_BASE}/nfl/v1/teams").mock(return_value=httpx.Response(200, json=_TEAMS))
    respx.get(f"{_BASE}/nfl/v1/team_season_stats").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={"data": [{"points": 420 if "team_ids%5B%5D=1" in str(request.url) else 310}]},
        )
    )
    respx.get(f"{_BASE}/nfl/v1/season_stats").mock(
        return_value=httpx.Response(200, json={"data": [{"player": {"id": 9}, "passing_yards": 4100}]})
    )
    state = {"sport": "americanfootball_nfl", "games": [_game()]}

    async with httpx.AsyncClient() as client:
        out = await stats_agent(state, client=client)

    assert "Kansas City Chiefs" in out["team_stats"]
    assert "Las Vegas Raiders" in out["team_stats"]
    assert out["player_stats"]["Kansas City Chiefs"] == [{"player": {"id": 9}, "passing_yards": 4100}]


@pytest.mark.asyncio
async def test_stats_agent_skips_unsupported_sport():
    state = {"sport": "baseball_mlb_playoffs", "games": [_game(sport="baseball_mlb_playoffs")]}
    async with httpx.AsyncClient() as client:
        out = await stats_agent(state, client=client)
    assert out == {"team_stats": {}, "player_stats": {}}


@pytest.mark.asyncio
async def test_stats_agent_returns_empty_with_no_games():
    async with httpx.AsyncClient() as client:
        out = await stats_agent({"sport": "americanfootball_nfl", "games": []}, client=client)
    assert out == {"team_stats": {}, "player_stats": {}}


@pytest.mark.asyncio
@respx.mock
async def test_stats_agent_survives_teams_lookup_failure():
    respx.get(f"{_BASE}/nfl/v1/teams").mock(return_value=httpx.Response(500))
    state = {"sport": "americanfootball_nfl", "games": [_game()]}
    async with httpx.AsyncClient() as client:
        out = await stats_agent(state, client=client)
    assert out == {"team_stats": {}, "player_stats": {}}


@pytest.mark.asyncio
@respx.mock
async def test_stats_agent_skips_unmatched_team_without_crashing():
    respx.get(f"{_BASE}/nfl/v1/teams").mock(
        return_value=httpx.Response(200, json={"data": [{"id": 1, "full_name": "Kansas City Chiefs"}]})
    )
    respx.get(f"{_BASE}/nfl/v1/team_season_stats").mock(
        return_value=httpx.Response(200, json={"data": [{"points": 420}]})
    )
    respx.get(f"{_BASE}/nfl/v1/season_stats").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    state = {"sport": "americanfootball_nfl", "games": [_game()]}  # Raiders won't resolve

    async with httpx.AsyncClient() as client:
        out = await stats_agent(state, client=client)

    assert "Kansas City Chiefs" in out["team_stats"]
    assert "Las Vegas Raiders" not in out["team_stats"]
