"""Tests for the injury agent: feed parsing and bounded adjustments."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from fairline.injuries import (
    injury_agent,
    injury_margin_adjustment,
    parse_espn_injuries,
)
from fairline.state import GameSnapshot

NOW = datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc)

_ESPN_PAYLOAD = {
    "injuries": [
        {
            "displayName": "Kansas City Chiefs",
            "injuries": [
                {
                    "athlete": {"displayName": "Patrick Mahomes", "position": {"abbreviation": "QB"}},
                    "status": "Out",
                },
                {
                    "athlete": {"displayName": "Some Receiver", "position": {"abbreviation": "WR"}},
                    "status": "Questionable",
                },
            ],
        },
        {
            "displayName": "Las Vegas Raiders",
            "injuries": [],
        },
    ]
}


def test_parse_espn_injuries_extracts_players_and_statuses():
    teams = parse_espn_injuries(_ESPN_PAYLOAD)

    kc = teams["Kansas City Chiefs"]
    assert {"player": "Patrick Mahomes", "position": "QB", "status": "Out"} in kc
    assert teams["Las Vegas Raiders"] == []


class TestInjuryAdjustment:
    def test_qb_out_is_the_big_one(self):
        adj, notes = injury_margin_adjustment(
            "americanfootball_nfl",
            [{"player": "Patrick Mahomes", "position": "QB", "status": "Out"}],
        )
        assert adj == pytest.approx(-5.5)
        assert "Patrick Mahomes (QB) Out" in notes

    def test_questionable_counts_half(self):
        adj, _ = injury_margin_adjustment(
            "americanfootball_nfl",
            [{"player": "Patrick Mahomes", "position": "QB", "status": "Questionable"}],
        )
        assert adj == pytest.approx(-2.75)

    def test_total_is_capped(self):
        wall = [
            {"player": f"P{i}", "position": "QB", "status": "Out"} for i in range(4)
        ]
        adj, _ = injury_margin_adjustment("americanfootball_nfl", wall)
        assert adj == -8.0

    def test_healthy_team_is_neutral(self):
        adj, notes = injury_margin_adjustment("americanfootball_nfl", [])
        assert adj == 0.0
        assert notes == []


@pytest.mark.asyncio
@respx.mock
async def test_injury_agent_attaches_adjustments_for_slate_teams():
    respx.get(url__startswith="https://site.api.espn.com/").mock(
        return_value=httpx.Response(200, json=_ESPN_PAYLOAD)
    )
    game = GameSnapshot(
        game_id="g1",
        sport="americanfootball_nfl",
        home_team="Kansas City Chiefs",
        away_team="Las Vegas Raiders",
        commence_time=NOW + timedelta(days=2),
        bookmakers=[],
    )
    async with httpx.AsyncClient() as client:
        out = await injury_agent({"sport": "americanfootball_nfl", "games": [game]}, client=client)

    kc = out["team_injuries"]["Kansas City Chiefs"]
    assert kc["adjustment"] == pytest.approx(-5.75)  # QB out plus half a WR
    assert any("Mahomes" in n for n in kc["notes"])
    assert "Las Vegas Raiders" not in out["team_injuries"]


@pytest.mark.asyncio
@respx.mock
async def test_injury_agent_survives_feed_failure():
    respx.get(url__startswith="https://site.api.espn.com/").mock(
        return_value=httpx.Response(500)
    )
    game = GameSnapshot(
        game_id="g1",
        sport="americanfootball_nfl",
        home_team="Kansas City Chiefs",
        away_team="Las Vegas Raiders",
        commence_time=NOW,
        bookmakers=[],
    )
    async with httpx.AsyncClient() as client:
        out = await injury_agent({"sport": "americanfootball_nfl", "games": [game]}, client=client)
    assert out == {"team_injuries": {}}
