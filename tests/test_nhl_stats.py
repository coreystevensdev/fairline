"""Tests for NHL ingestion: rest-day derivation, boxscore-to-row parsing."""

from __future__ import annotations

import httpx
import pytest
import respx

from fairline.nhl_stats import _derive_rest_days, fetch_nhl_skater_games

_BASE = "https://api-web.nhle.com"

_TEAM_NAMES = {"EDM": "Edmonton Oilers", "CGY": "Calgary Flames"}


class TestDeriveRestDays:
    def test_first_game_in_range_has_no_rest_days(self):
        games = [{"game_date": "2025-12-01"}]
        rest = _derive_rest_days(games)
        assert rest[0] is None

    def test_second_game_computes_day_difference(self):
        games = [{"game_date": "2025-12-01"}, {"game_date": "2025-12-03"}]
        rest = _derive_rest_days(games)
        assert rest[1] == 2

    def test_back_to_back_is_one_day_rest(self):
        games = [{"game_date": "2025-12-01"}, {"game_date": "2025-12-02"}]
        rest = _derive_rest_days(games)
        assert rest[1] == 1


@pytest.mark.asyncio
@respx.mock
async def test_fetch_nhl_skater_games_happy_path(monkeypatch):
    monkeypatch.setattr("fairline.nhl_stats._TEAM_NAMES", _TEAM_NAMES)
    respx.get(f"{_BASE}/v1/club-schedule-season/EDM/20252026").mock(
        return_value=httpx.Response(200, json={
            "games": [{"id": 1, "gameDate": "2025-12-01", "homeTeam": {"abbrev": "EDM"}, "awayTeam": {"abbrev": "CGY"}}]
        })
    )
    respx.get(f"{_BASE}/v1/gamecenter/1/boxscore").mock(
        return_value=httpx.Response(200, json={
            "playerByGameStats": {
                "homeTeam": {
                    "forwards": [{"playerId": 1, "name": {"default": "Connor McDavid"}, "position": "C", "goals": 1, "assists": 2, "points": 3, "sog": 5}],
                    "defense": [],
                    "goalies": [{"playerId": 2, "name": {"default": "Dustin Wolf"}, "starter": False}],
                },
                "awayTeam": {
                    "forwards": [],
                    "defense": [],
                    "goalies": [{"playerId": 3, "name": {"default": "Jacob Markstrom"}, "starter": True}],
                },
            }
        })
    )
    async with httpx.AsyncClient() as client:
        rows = await fetch_nhl_skater_games(client, "EDM", "20252026")

    assert len(rows) == 1
    row = rows[0]
    assert row.player == "Connor McDavid"
    assert row.team == "Edmonton Oilers"
    assert row.opponent == "Calgary Flames"
    assert row.is_home is True
    assert row.opposing_goalie == "Jacob Markstrom"
    assert row.goals == 1
    assert row.points == 3
    assert row.rest_days is None  # only game in range
