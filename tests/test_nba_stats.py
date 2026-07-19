"""Tests for NBA ingestion: MATCHUP parsing, rest-day derivation, row construction."""

from __future__ import annotations

import pytest

from fairline.nba_stats import _derive_rest_days, _parse_matchup, fetch_nba_player_games

_TEAM_NAMES = {"LAL": "Los Angeles Lakers", "BOS": "Boston Celtics"}


class TestParseMatchup:
    def test_home_game_format(self):
        is_home, opponent_code = _parse_matchup("LAL vs. BOS")
        assert is_home is True
        assert opponent_code == "BOS"

    def test_away_game_format(self):
        is_home, opponent_code = _parse_matchup("LAL @ BOS")
        assert is_home is False
        assert opponent_code == "BOS"


class TestDeriveRestDays:
    def test_first_game_has_no_rest_days(self):
        games = [{"GAME_DATE": "2025-12-01"}]
        rest = _derive_rest_days(games)
        assert rest[0] is None

    def test_back_to_back_is_one_day_rest(self):
        games = [{"GAME_DATE": "2025-12-01"}, {"GAME_DATE": "2025-12-02"}]
        rest = _derive_rest_days(games)
        assert rest[1] == 1


@pytest.mark.asyncio
async def test_fetch_nba_player_games_happy_path(monkeypatch):
    rows = [
        {
            "PLAYER_NAME": "LeBron James", "TEAM_ABBREVIATION": "LAL", "MATCHUP": "LAL vs. BOS",
            "GAME_DATE": "2025-12-01", "PTS": 28, "REB": 8, "AST": 9, "FG3M": 3,
        },
        {
            "PLAYER_NAME": "LeBron James", "TEAM_ABBREVIATION": "LAL", "MATCHUP": "LAL @ BOS",
            "GAME_DATE": "2025-12-03", "PTS": 22, "REB": 6, "AST": 7, "FG3M": 1,
        },
    ]

    async def fake_fetch(season, proxy=None):
        return rows

    async def fake_positions(season, proxy=None):
        return {}

    monkeypatch.setattr("fairline.nba_stats.fetch_league_game_log", fake_fetch)
    monkeypatch.setattr("fairline.nba_stats.fetch_league_positions", fake_positions)
    monkeypatch.setattr("fairline.nba_stats._TEAM_NAMES", _TEAM_NAMES)

    result = await fetch_nba_player_games("2024-25")

    assert len(result) == 2
    first, second = sorted(result, key=lambda g: g.game_date)
    assert first.player == "LeBron James"
    assert first.team == "Los Angeles Lakers"
    assert first.opponent == "Boston Celtics"
    assert first.is_home is True
    assert first.rest_days is None
    assert first.points == 28
    assert second.is_home is False
    assert second.rest_days == 2


@pytest.mark.asyncio
async def test_fetch_nba_player_games_joins_position(monkeypatch):
    rows = [
        {
            "PLAYER_NAME": "LeBron James", "TEAM_ABBREVIATION": "LAL", "MATCHUP": "LAL vs. BOS",
            "GAME_DATE": "2025-12-01", "PTS": 28, "REB": 8, "AST": 9, "FG3M": 3,
        },
    ]

    async def fake_fetch(season, proxy=None):
        return rows

    async def fake_positions(season, proxy=None):
        return {"LeBron James": "Forward"}

    monkeypatch.setattr("fairline.nba_stats.fetch_league_game_log", fake_fetch)
    monkeypatch.setattr("fairline.nba_stats.fetch_league_positions", fake_positions)
    monkeypatch.setattr("fairline.nba_stats._TEAM_NAMES", {"LAL": "Los Angeles Lakers", "BOS": "Boston Celtics"})

    result = await fetch_nba_player_games("2024-25")

    assert result[0].position == "Forward"


@pytest.mark.asyncio
async def test_fetch_nba_player_games_leaves_position_null_when_unmatched(monkeypatch):
    rows = [
        {
            "PLAYER_NAME": "Unknown Player", "TEAM_ABBREVIATION": "LAL", "MATCHUP": "LAL vs. BOS",
            "GAME_DATE": "2025-12-01", "PTS": 10, "REB": 2, "AST": 1, "FG3M": 0,
        },
    ]

    async def fake_fetch(season, proxy=None):
        return rows

    async def fake_positions(season, proxy=None):
        return {"LeBron James": "Forward"}  # no entry for "Unknown Player"

    monkeypatch.setattr("fairline.nba_stats.fetch_league_game_log", fake_fetch)
    monkeypatch.setattr("fairline.nba_stats.fetch_league_positions", fake_positions)
    monkeypatch.setattr("fairline.nba_stats._TEAM_NAMES", {"LAL": "Los Angeles Lakers", "BOS": "Boston Celtics"})

    result = await fetch_nba_player_games("2024-25")

    assert result[0].position is None
