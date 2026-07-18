"""Integration tests for the NHL API client using respx HTTP mocking."""

from __future__ import annotations

import httpx
import pytest
import respx

from fairline.clients.nhl_api import fetch_boxscore, fetch_team_schedule

_BASE = "https://api-web.nhle.com"

_SCHEDULE_FIXTURE = {
    "games": [
        {"id": 2025020123, "gameDate": "2025-12-01", "homeTeam": {"abbrev": "EDM"}, "awayTeam": {"abbrev": "CGY"}, "gameState": "OFF"},
        {"id": 2025020145, "gameDate": "2025-12-03", "homeTeam": {"abbrev": "CGY"}, "awayTeam": {"abbrev": "EDM"}, "gameState": "FUT"},
    ]
}


@pytest.mark.asyncio
@respx.mock
async def test_fetch_team_schedule_happy_path():
    respx.get(f"{_BASE}/v1/club-schedule-season/EDM/20252026").mock(
        return_value=httpx.Response(200, json=_SCHEDULE_FIXTURE)
    )
    async with httpx.AsyncClient() as client:
        games = await fetch_team_schedule(client, "EDM", "20252026")
    assert len(games) == 2
    assert games[0]["game_id"] == 2025020123
    assert games[0]["home_team"] == "EDM"
    assert games[0]["away_team"] == "CGY"
    assert games[0]["game_date"] == "2025-12-01"
    assert games[0]["game_state"] == "OFF"
    assert games[1]["game_state"] == "FUT"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_boxscore_happy_path():
    boxscore_fixture = {
        "playerByGameStats": {
            "homeTeam": {
                "forwards": [{"playerId": 8478402, "name": {"default": "Connor McDavid"}, "position": "C", "goals": 1, "assists": 2, "points": 3, "sog": 5}],
                "defense": [],
                "goalies": [{"playerId": 8480313, "name": {"default": "Dustin Wolf"}, "starter": True}],
            },
            "awayTeam": {"forwards": [], "defense": [], "goalies": []},
        }
    }
    respx.get(f"{_BASE}/v1/gamecenter/2025020123/boxscore").mock(
        return_value=httpx.Response(200, json=boxscore_fixture)
    )
    async with httpx.AsyncClient() as client:
        data = await fetch_boxscore(client, 2025020123)
    assert data["playerByGameStats"]["homeTeam"]["forwards"][0]["goals"] == 1
