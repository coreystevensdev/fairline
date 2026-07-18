"""NHL's own free official API: no auth, no key, plain JSON.

https://api-web.nhle.com has no official OpenAPI spec, but every field path
used here was checked against a real live response (club-schedule-season and
gamecenter/boxscore) during planning, not assumed from third-party docs.
"""

from __future__ import annotations

import httpx

_BASE = "https://api-web.nhle.com"


async def fetch_team_schedule(client: httpx.AsyncClient, team: str, season: str) -> list[dict]:
    """A team's full season schedule: game_id, game_date, home_team, away_team.

    `team` is the 3-letter abbreviation (e.g. "EDM"), `season` is the 8-digit
    season code (e.g. "20252026" for the 2025-26 season).
    """
    resp = await client.get(f"{_BASE}/v1/club-schedule-season/{team}/{season}", timeout=httpx.Timeout(30.0))
    resp.raise_for_status()
    payload = resp.json()
    games = []
    for g in payload.get("games") or []:
        games.append({
            "game_id": g["id"],
            "game_date": g["gameDate"],
            "home_team": g["homeTeam"]["abbrev"],
            "away_team": g["awayTeam"]["abbrev"],
            "game_state": g.get("gameState"),
        })
    return games


async def fetch_boxscore(client: httpx.AsyncClient, game_id: int) -> dict:
    """One game's full box score, raw parsed JSON (skater and goalie stat lines
    for both teams, nested under playerByGameStats)."""
    resp = await client.get(f"{_BASE}/v1/gamecenter/{game_id}/boxscore", timeout=httpx.Timeout(30.0))
    resp.raise_for_status()
    return resp.json()
