"""Integration tests for the MLB schedule client using respx HTTP mocking."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from fairline.clients.mlb_schedule_client import fetch_probable_pitchers

_BASE = "https://statsapi.mlb.com"

_SCHEDULE_FIXTURE = {
    "dates": [
        {
            "games": [
                {
                    "gameDate": "2026-07-19T16:15:00Z",
                    "teams": {
                        "home": {
                            "team": {"name": "Toronto Blue Jays"},
                            "probablePitcher": {"fullName": "Trey Yesavage"},
                        },
                        "away": {
                            "team": {"name": "Chicago White Sox"},
                            "probablePitcher": {"fullName": "Sean Burke"},
                        },
                    },
                },
                {
                    "gameDate": "2026-07-19T20:07:00Z",
                    "teams": {
                        "home": {"team": {"name": "Los Angeles Angels"}},
                        "away": {"team": {"name": "Detroit Tigers"}},
                    },
                },
            ]
        }
    ]
}


@pytest.mark.asyncio
@respx.mock
async def test_fetch_probable_pitchers_happy_path():
    respx.get(f"{_BASE}/api/v1/schedule", params={"sportId": "1", "date": "2026-07-19", "hydrate": "probablePitcher"}).mock(
        return_value=httpx.Response(200, json=_SCHEDULE_FIXTURE)
    )
    async with httpx.AsyncClient() as client:
        games = await fetch_probable_pitchers(client, "2026-07-19")

    assert len(games) == 2
    assert games[0]["home_team"] == "Toronto Blue Jays"
    assert games[0]["away_team"] == "Chicago White Sox"
    assert games[0]["commence_time"] == datetime(2026, 7, 19, 16, 15, tzinfo=timezone.utc)
    assert games[0]["home_pitcher"] == "Trey Yesavage"
    assert games[0]["away_pitcher"] == "Sean Burke"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_probable_pitchers_missing_probable_is_none():
    respx.get(f"{_BASE}/api/v1/schedule", params={"sportId": "1", "date": "2026-07-19", "hydrate": "probablePitcher"}).mock(
        return_value=httpx.Response(200, json=_SCHEDULE_FIXTURE)
    )
    async with httpx.AsyncClient() as client:
        games = await fetch_probable_pitchers(client, "2026-07-19")

    # Angels/Tigers game has no probablePitcher key on either side yet
    # (common the day before a start is announced) -- must be None, not KeyError.
    assert games[1]["home_pitcher"] is None
    assert games[1]["away_pitcher"] is None


@pytest.mark.asyncio
@respx.mock
async def test_fetch_probable_pitchers_empty_date_returns_empty_list():
    respx.get(f"{_BASE}/api/v1/schedule", params={"sportId": "1", "date": "2026-01-01", "hydrate": "probablePitcher"}).mock(
        return_value=httpx.Response(200, json={"dates": []})
    )
    async with httpx.AsyncClient() as client:
        games = await fetch_probable_pitchers(client, "2026-01-01")
    assert games == []
