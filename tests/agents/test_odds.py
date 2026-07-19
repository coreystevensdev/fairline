"""Unit tests for odds_agent node logic.

Tests cover best_sharp_book priority ordering and _derive_fair_line math.
These are pure (no network, no graph) so they run without API keys.
"""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest

from fairline.agents.odds import best_sharp_book, odds_agent, _derive_fair_line
from fairline.state import (
    BookmakerOdds,
    GameSnapshot,
    MarketOdds,
    Outcome,
)


def _make_game(bookmaker_keys: list[str], market_prices: list[int] | None = None) -> GameSnapshot:
    """Helper: build a minimal GameSnapshot with the given bookmaker keys."""
    if market_prices is None:
        market_prices = [-110, -110]
    bookmakers = []
    for key in bookmaker_keys:
        outcomes = [
            Outcome(name="Team A", price=market_prices[0]),
            Outcome(name="Team B", price=market_prices[1]),
        ]
        mkt = MarketOdds(key="spreads", outcomes=outcomes)
        bookmakers.append(BookmakerOdds(key=key, title=key.capitalize(), markets=[mkt]))
    return GameSnapshot(
        game_id="test-game-1",
        sport="americanfootball_nfl",
        home_team="Team A",
        away_team="Team B",
        commence_time=datetime(2026, 1, 15, 20, 0),
        bookmakers=bookmakers,
    )


class TestBestSharpBook:
    def test_pinnacle_wins_over_betonline(self):
        game = _make_game(["betonlineag", "pinnacle", "fanduel"])
        assert best_sharp_book(game) == "pinnacle"

    def test_betonline_wins_when_pinnacle_absent(self):
        game = _make_game(["mybookieag", "betonlineag", "fanduel"])
        assert best_sharp_book(game) == "betonlineag"

    def test_mybookie_last_resort(self):
        game = _make_game(["mybookieag", "draftkings"])
        assert best_sharp_book(game) == "mybookieag"

    def test_no_sharp_book_returns_none(self):
        game = _make_game(["fanduel", "draftkings", "betmgm"])
        assert best_sharp_book(game) is None

    def test_only_pinnacle_present(self):
        game = _make_game(["pinnacle"])
        assert best_sharp_book(game) == "pinnacle"


class TestDeriveFairLine:
    def test_even_spread_produces_fifty_fifty(self):
        game = _make_game(["pinnacle"], market_prices=[-110, -110])
        fl = _derive_fair_line(game, "spreads", "pinnacle")
        assert fl is not None
        assert abs(fl.fair_probs[0] - 0.5) < 0.001
        assert abs(fl.fair_probs[1] - 0.5) < 0.001
        assert abs(sum(fl.fair_probs) - 1.0) < 1e-9

    def test_favorite_gets_higher_fair_probability(self):
        # -200 favorite, +175 dog
        game = _make_game(["pinnacle"], market_prices=[-200, 175])
        fl = _derive_fair_line(game, "spreads", "pinnacle")
        assert fl is not None
        assert fl.fair_probs[0] > fl.fair_probs[1]

    def test_fair_probs_sum_to_one(self):
        game = _make_game(["pinnacle"], market_prices=[-150, 130])
        fl = _derive_fair_line(game, "spreads", "pinnacle")
        assert fl is not None
        assert abs(sum(fl.fair_probs) - 1.0) < 1e-9

    def test_missing_book_returns_none(self):
        game = _make_game(["fanduel"])
        fl = _derive_fair_line(game, "spreads", "pinnacle")
        assert fl is None

    def test_missing_market_returns_none(self):
        game = _make_game(["pinnacle"])
        # Game only has "spreads" market; asking for "h2h" should fail.
        fl = _derive_fair_line(game, "h2h", "pinnacle")
        assert fl is None

    def test_fair_line_metadata(self):
        game = _make_game(["pinnacle"])
        fl = _derive_fair_line(game, "spreads", "pinnacle")
        assert fl is not None
        assert fl.game_id == "test-game-1"
        assert fl.market == "spreads"
        assert fl.source_book == "pinnacle"
        assert fl.outcomes == ["Team A", "Team B"]


class TestOddsAgentErrorClassification:
    """odds_agent must route to the same short-circuit on any odds-fetch
    failure, but the error string should name which failure class occurred
    instead of flattening auth, quota, network, and parse failures into one
    generic message."""

    @staticmethod
    def _http_status_error(status: int) -> httpx.HTTPStatusError:
        request = httpx.Request("GET", "https://api.the-odds-api.com/v4/sports/x/odds/")
        response = httpx.Response(status, request=request)
        return httpx.HTTPStatusError(f"{status}", request=request, response=response)

    @pytest.mark.asyncio
    async def test_auth_error_is_named(self, monkeypatch):
        async def fake_fetch_odds(client, sport):
            raise self._http_status_error(401)

        monkeypatch.setattr("fairline.agents.odds.fetch_odds", fake_fetch_odds)
        result = await odds_agent({"sport": "americanfootball_nfl"}, client=None)
        assert "auth" in result["error"].lower()
        assert result["games"] == []
        assert result["fair_lines"] == []

    @pytest.mark.asyncio
    async def test_quota_error_is_named(self, monkeypatch):
        async def fake_fetch_odds(client, sport):
            raise self._http_status_error(429)

        monkeypatch.setattr("fairline.agents.odds.fetch_odds", fake_fetch_odds)
        result = await odds_agent({"sport": "americanfootball_nfl"}, client=None)
        assert "quota" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_network_error_is_named(self, monkeypatch):
        async def fake_fetch_odds(client, sport):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr("fairline.agents.odds.fetch_odds", fake_fetch_odds)
        result = await odds_agent({"sport": "americanfootball_nfl"}, client=None)
        assert "network" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_timeout_is_named(self, monkeypatch):
        async def fake_fetch_odds(client, sport):
            raise httpx.ReadTimeout("timed out")

        monkeypatch.setattr("fairline.agents.odds.fetch_odds", fake_fetch_odds)
        result = await odds_agent({"sport": "americanfootball_nfl"}, client=None)
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_parse_error_is_named(self, monkeypatch):
        async def fake_fetch_odds(client, sport):
            raise json.JSONDecodeError("Expecting value", "not json", 0)

        monkeypatch.setattr("fairline.agents.odds.fetch_odds", fake_fetch_odds)
        result = await odds_agent({"sport": "americanfootball_nfl"}, client=None)
        assert "malformed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_api_key_is_named_as_configuration(self, monkeypatch):
        async def fake_fetch_odds(client, sport):
            raise RuntimeError("ODDS_API_KEY is not set")

        monkeypatch.setattr("fairline.agents.odds.fetch_odds", fake_fetch_odds)
        result = await odds_agent({"sport": "americanfootball_nfl"}, client=None)
        assert "configuration" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_every_failure_class_still_short_circuits_with_an_error(self, monkeypatch):
        for exc in (
            self._http_status_error(500),
            httpx.ConnectError("boom"),
            json.JSONDecodeError("bad", "doc", 0),
            RuntimeError("no key"),
            ValueError("unsupported sport"),
        ):
            async def fake_fetch_odds(client, sport, _exc=exc):
                raise _exc

            monkeypatch.setattr("fairline.agents.odds.fetch_odds", fake_fetch_odds)
            result = await odds_agent({"sport": "americanfootball_nfl"}, client=None)
            assert result["error"]
            assert result["games"] == []
            assert result["fair_lines"] == []
