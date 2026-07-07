"""Weather context for outdoor NFL games.

Wind is the one weather input with a well-documented totals effect; the
adjustment is bounded in code and every reading lands in the pick rationale
so the reviewer sees why a number moved. Open-Meteo is free, keyless, and
forecasts 16 days out; games beyond the horizon or under a roof are skipped.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from fairline.state import FairlineState

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_HORIZON_DAYS = 16

# City-level coordinates per NFL home team; stadium precision buys nothing at
# forecast resolution.
STADIUMS = {
    "Arizona Cardinals": (33.5, -112.3), "Atlanta Falcons": (33.8, -84.4),
    "Baltimore Ravens": (39.3, -76.6), "Buffalo Bills": (42.8, -78.8),
    "Carolina Panthers": (35.2, -80.9), "Chicago Bears": (41.9, -87.6),
    "Cincinnati Bengals": (39.1, -84.5), "Cleveland Browns": (41.5, -81.7),
    "Dallas Cowboys": (32.7, -97.1), "Denver Broncos": (39.7, -105.0),
    "Detroit Lions": (42.3, -83.0), "Green Bay Packers": (44.5, -88.1),
    "Houston Texans": (29.7, -95.4), "Indianapolis Colts": (39.8, -86.2),
    "Jacksonville Jaguars": (30.3, -81.6), "Kansas City Chiefs": (39.0, -94.5),
    "Las Vegas Raiders": (36.1, -115.2), "Los Angeles Chargers": (33.9, -118.3),
    "Los Angeles Rams": (33.9, -118.3), "Miami Dolphins": (25.9, -80.2),
    "Minnesota Vikings": (44.9, -93.2), "New England Patriots": (42.1, -71.3),
    "New Orleans Saints": (30.0, -90.1), "New York Giants": (40.8, -74.1),
    "New York Jets": (40.8, -74.1), "Philadelphia Eagles": (39.9, -75.2),
    "Pittsburgh Steelers": (40.4, -80.0), "San Francisco 49ers": (37.4, -121.9),
    "Seattle Seahawks": (47.6, -122.3), "Tampa Bay Buccaneers": (28.0, -82.5),
    "Tennessee Titans": (36.2, -86.8), "Washington Commanders": (38.9, -76.9),
}

DOMES = {
    "Arizona Cardinals", "Atlanta Falcons", "Dallas Cowboys", "Detroit Lions",
    "Houston Texans", "Indianapolis Colts", "Las Vegas Raiders",
    "Los Angeles Chargers", "Los Angeles Rams", "Minnesota Vikings",
    "New Orleans Saints",
}


def wind_total_adjustment(wind_mph: float) -> float:
    """Points off the expected total: roughly a third of a point per mph over 10.

    Capped at -7; beyond that the game script changes in ways a linear
    adjustment cannot honestly claim to model.
    """
    return max(-7.0, -0.35 * max(0.0, wind_mph - 10.0))


async def weather_agent(state: FairlineState, client: httpx.AsyncClient) -> dict:
    """Attach kickoff-hour forecasts for outdoor NFL games."""
    games = state.get("games", [])
    if state.get("sport") != "americanfootball_nfl" or not games:
        return {"game_weather": {}}

    now = datetime.now(timezone.utc)
    weather: dict = {}
    for game in games:
        coords = STADIUMS.get(game.home_team)
        if coords is None or game.home_team in DOMES:
            continue
        kickoff = game.commence_time
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        if kickoff - now > timedelta(days=FORECAST_HORIZON_DAYS):
            continue
        try:
            resp = await client.get(
                FORECAST_URL,
                params={
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
                    "wind_speed_unit": "mph",
                    "temperature_unit": "fahrenheit",
                    "timezone": "UTC",
                    "forecast_days": FORECAST_HORIZON_DAYS,
                },
                timeout=httpx.Timeout(15.0),
            )
            resp.raise_for_status()
            hourly = resp.json().get("hourly") or {}
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("weather: fetch failed for %s: %s", game.home_team, exc)
            continue

        times = hourly.get("time") or []
        if not times:
            continue
        stamps = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times]
        idx = min(range(len(stamps)), key=lambda i: abs(stamps[i] - kickoff))

        def at(key):
            values = hourly.get(key) or []
            return values[idx] if idx < len(values) else None

        wind = at("wind_speed_10m")
        if wind is None:
            continue
        weather[game.game_id] = {
            "wind_mph": float(wind),
            "temp_f": float(at("temperature_2m")) if at("temperature_2m") is not None else None,
            "precip_prob": at("precipitation_probability"),
        }

    logger.info("weather_agent: forecasts for %d of %d games", len(weather), len(games))
    return {"game_weather": weather}
