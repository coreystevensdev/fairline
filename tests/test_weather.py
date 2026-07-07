"""Tests for the weather agent: forecasts for outdoor games, bounded totals impact."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from fairline.state import GameSnapshot
from fairline.weather import DOMES, wind_total_adjustment, weather_agent

NOW = datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc)


def _game(home="Kansas City Chiefs", game_id="g1", kickoff=None) -> GameSnapshot:
    return GameSnapshot(
        game_id=game_id,
        sport="americanfootball_nfl",
        home_team=home,
        away_team="Las Vegas Raiders",
        commence_time=kickoff or (NOW + timedelta(days=2)),
        bookmakers=[],
    )


class TestWindAdjustment:
    def test_calm_air_changes_nothing(self):
        assert wind_total_adjustment(8.0) == 0.0

    def test_wind_over_ten_costs_points(self):
        # 20 mph: -0.35 * 10 = -3.5 points off the expected total
        assert wind_total_adjustment(20.0) == pytest.approx(-3.5)

    def test_adjustment_is_capped(self):
        assert wind_total_adjustment(60.0) == -7.0


def _forecast_payload(kickoff: datetime, wind: float) -> dict:
    hours = [kickoff + timedelta(hours=h - 2) for h in range(5)]
    return {
        "hourly": {
            "time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
            "wind_speed_10m": [5.0, 5.0, wind, 5.0, 5.0],
            "temperature_2m": [30.0] * 5,
            "precipitation_probability": [10] * 5,
        }
    }


@pytest.mark.asyncio
@respx.mock
async def test_weather_agent_reads_nearest_hour_for_outdoor_game():
    kickoff = NOW + timedelta(days=2)
    respx.get(url__startswith="https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=_forecast_payload(kickoff, 22.0))
    )
    state = {"sport": "americanfootball_nfl", "games": [_game(kickoff=kickoff)]}

    async with httpx.AsyncClient() as client:
        out = await weather_agent(state, client=client)

    wx = out["game_weather"]["g1"]
    assert wx["wind_mph"] == 22.0
    assert wx["temp_f"] == 30.0


@pytest.mark.asyncio
async def test_weather_agent_skips_domes_and_other_sports():
    dome_home = next(iter(DOMES))
    state = {
        "sport": "americanfootball_nfl",
        "games": [_game(home=dome_home)],
    }
    async with httpx.AsyncClient() as client:
        out = await weather_agent(state, client=client)
    assert out == {"game_weather": {}}

    async with httpx.AsyncClient() as client:
        out = await weather_agent({"sport": "basketball_nba", "games": [_game()]}, client=client)
    assert out == {"game_weather": {}}


@pytest.mark.asyncio
@respx.mock
async def test_weather_agent_survives_a_failed_fetch():
    respx.get(url__startswith="https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(500)
    )
    state = {"sport": "americanfootball_nfl", "games": [_game()]}
    async with httpx.AsyncClient() as client:
        out = await weather_agent(state, client=client)
    assert out == {"game_weather": {}}
