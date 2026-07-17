"""BALLDONTLIE client: team and player season stats across four sports.

Free tier: 5 requests/minute, key via a plain `Authorization: <key>` header
(no "Bearer" prefix). BALLDONTLIE's team IDs are internal to that API, they
have no relation to the team-name strings fairline gets from The Odds API,
so every stats lookup starts by resolving a name against that sport's
fetched teams list.

Each sport's season-stats endpoints have a different shape (NFL/MLB accept
a `team_id` filter directly; NHL is a per-team path parameter; NBA needs a
`season_type` alongside `season`), so `fetch_team_season_stats` and
`fetch_player_season_stats` branch per sport rather than pretending one
call signature fits all four.

Docs: https://docs.balldontlie.io/
"""

from __future__ import annotations

import os

import httpx

_BASE = "https://api.balldontlie.io"

SPORT_PATH = {
    "americanfootball_nfl": "nfl",
    "basketball_nba": "nba",
    "baseball_mlb": "mlb",
    "icehockey_nhl": "nhl",
}

# NBA's and NHL's player endpoints don't accept a team_id filter; NBA wants
# player_ids, NHL is single-player-by-path only. Roster resolution to get
# those player IDs is separate future work, not this feature.
PLAYER_STATS_SUPPORTED = {"nfl", "mlb"}


def _api_key() -> str:
    key = os.environ.get("BALLDONTLIE_API_KEY", "")
    if not key:
        raise RuntimeError("BALLDONTLIE_API_KEY is not set")
    return key


async def _get(client: httpx.AsyncClient, path: str, params: dict | None = None) -> dict:
    resp = await client.get(
        f"{_BASE}{path}",
        params=params or {},
        headers={"Authorization": _api_key()},
        timeout=httpx.Timeout(15.0),
    )
    resp.raise_for_status()
    return resp.json()


async def fetch_teams(client: httpx.AsyncClient, sport: str) -> list[dict]:
    """All teams for one BALLDONTLIE sport path segment (e.g. "nfl", "mlb")."""
    payload = await _get(client, f"/{sport}/v1/teams")
    return payload.get("data") or []


def resolve_team_id(teams: list[dict], team_full_name: str) -> int | None:
    """Match an Odds-API team name (e.g. "Kansas City Chiefs") to a BALLDONTLIE team id."""
    target = team_full_name.strip().lower()
    for team in teams:
        if (team.get("full_name") or "").strip().lower() == target:
            return team.get("id")
    return None


async def fetch_team_season_stats(
    client: httpx.AsyncClient, sport: str, team_id: int, season: int
) -> dict | None:
    """One team's season stats, or None when the sport has nothing for that team/season."""
    if sport == "nfl":
        payload = await _get(
            client, "/nfl/v1/team_season_stats", {"team_ids[]": team_id, "season": season}
        )
    elif sport == "mlb":
        payload = await _get(
            client, "/mlb/v1/teams/season_stats", {"team_id": team_id, "season": season}
        )
    elif sport == "nhl":
        payload = await _get(client, f"/nhl/v1/teams/{team_id}/season_stats", {"season": season})
        rows = payload.get("data") or []
        return {row["name"]: row["value"] for row in rows} if rows else None
    elif sport == "nba":
        payload = await _get(
            client,
            "/nba/v1/team_season_averages/general",
            {"team_ids[]": team_id, "season": season, "season_type": "regular"},
        )
    else:
        raise ValueError(f"unsupported sport {sport!r}")

    rows = payload.get("data") or []
    return rows[0] if rows else None


async def fetch_player_season_stats(
    client: httpx.AsyncClient, sport: str, team_id: int, season: int
) -> list[dict]:
    """Player season stats for every player on one team. NFL and MLB only."""
    if sport not in PLAYER_STATS_SUPPORTED:
        return []
    path = "/nfl/v1/season_stats" if sport == "nfl" else "/mlb/v1/season_stats"
    payload = await _get(client, path, {"team_id": team_id, "season": season})
    return payload.get("data") or []
