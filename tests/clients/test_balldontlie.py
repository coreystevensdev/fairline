"""Integration tests for the BALLDONTLIE client using respx HTTP mocking.

All tests intercept at the httpx transport layer; BALLDONTLIE_API_KEY is set
to a fake value via monkeypatch.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from fairline.clients.balldontlie import (
    SPORT_PATH,
    fetch_player_season_stats,
    fetch_team_season_stats,
    fetch_teams,
    resolve_team_id,
)

_FAKE_KEY = "test-bdl-key"
_BASE = "https://api.balldontlie.io"

_NFL_TEAMS = {
    "data": [
        {"id": 1, "conference": "AFC", "division": "WEST", "location": "Kansas City",
         "name": "Chiefs", "full_name": "Kansas City Chiefs", "abbreviation": "KC"},
        {"id": 2, "conference": "AFC", "division": "WEST", "location": "Las Vegas",
         "name": "Raiders", "full_name": "Las Vegas Raiders", "abbreviation": "LV"},
    ]
}


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("BALLDONTLIE_API_KEY", _FAKE_KEY)


def test_sport_path_covers_all_four_sports():
    assert SPORT_PATH == {
        "americanfootball_nfl": "nfl",
        "basketball_nba": "nba",
        "baseball_mlb": "mlb",
        "icehockey_nhl": "nhl",
    }


class TestResolveTeamId:
    def test_exact_full_name_match(self):
        assert resolve_team_id(_NFL_TEAMS["data"], "Kansas City Chiefs") == 1

    def test_case_insensitive_match(self):
        assert resolve_team_id(_NFL_TEAMS["data"], "kansas city chiefs") == 2 - 1

    def test_no_match_returns_none(self):
        assert resolve_team_id(_NFL_TEAMS["data"], "Denver Broncos") is None


@pytest.mark.asyncio
@respx.mock
async def test_fetch_teams_happy_path():
    respx.get(f"{_BASE}/nfl/v1/teams").mock(
        return_value=httpx.Response(200, json=_NFL_TEAMS)
    )
    async with httpx.AsyncClient() as client:
        teams = await fetch_teams(client, "nfl")
    assert len(teams) == 2
    assert teams[0]["full_name"] == "Kansas City Chiefs"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_teams_sends_auth_header():
    route = respx.get(f"{_BASE}/nfl/v1/teams").mock(
        return_value=httpx.Response(200, json=_NFL_TEAMS)
    )
    async with httpx.AsyncClient() as client:
        await fetch_teams(client, "nfl")
    assert route.calls.last.request.headers["Authorization"] == _FAKE_KEY


@pytest.mark.asyncio
@respx.mock
async def test_fetch_team_season_stats_nfl():
    respx.get(f"{_BASE}/nfl/v1/team_season_stats").mock(
        return_value=httpx.Response(200, json={"data": [{"team": {"id": 1}, "points": 420}]})
    )
    async with httpx.AsyncClient() as client:
        stats = await fetch_team_season_stats(client, "nfl", team_id=1, season=2025)
    assert stats == {"team": {"id": 1}, "points": 420}


@pytest.mark.asyncio
@respx.mock
async def test_fetch_team_season_stats_nhl_uses_path_param():
    respx.get(f"{_BASE}/nhl/v1/teams/7/season_stats").mock(
        return_value=httpx.Response(200, json={"data": [{"name": "wins", "value": 41}]})
    )
    async with httpx.AsyncClient() as client:
        stats = await fetch_team_season_stats(client, "nhl", team_id=7, season=2025)
    assert stats == {"wins": 41}


@pytest.mark.asyncio
@respx.mock
async def test_fetch_team_season_stats_nba():
    respx.get(f"{_BASE}/nba/v1/team_season_averages/general").mock(
        return_value=httpx.Response(
            200, json={"data": [{"team": {"id": 5}, "off_rating": 118.2}]}
        )
    )
    async with httpx.AsyncClient() as client:
        stats = await fetch_team_season_stats(client, "nba", team_id=5, season=2025)
    assert stats == {"team": {"id": 5}, "off_rating": 118.2}


@pytest.mark.asyncio
async def test_fetch_team_season_stats_unsupported_sport_raises():
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError):
            await fetch_team_season_stats(client, "mls", team_id=1, season=2025)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_team_season_stats_empty_response_returns_none():
    respx.get(f"{_BASE}/mlb/v1/teams/season_stats").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    async with httpx.AsyncClient() as client:
        stats = await fetch_team_season_stats(client, "mlb", team_id=3, season=2025)
    assert stats is None


@pytest.mark.asyncio
@respx.mock
async def test_fetch_player_season_stats_nfl():
    respx.get(f"{_BASE}/nfl/v1/season_stats").mock(
        return_value=httpx.Response(200, json={"data": [{"player": {"id": 9}, "passing_yards": 4100}]})
    )
    async with httpx.AsyncClient() as client:
        players = await fetch_player_season_stats(client, "nfl", team_id=1, season=2025)
    assert players == [{"player": {"id": 9}, "passing_yards": 4100}]


@pytest.mark.asyncio
async def test_fetch_player_season_stats_unsupported_sport_returns_empty():
    async with httpx.AsyncClient() as client:
        players = await fetch_player_season_stats(client, "nba", team_id=1, season=2025)
    assert players == []


@pytest.mark.asyncio
@respx.mock
async def test_fetch_teams_missing_key_raises():
    import os

    os.environ.pop("BALLDONTLIE_API_KEY", None)
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError):
            await fetch_teams(client, "nfl")
