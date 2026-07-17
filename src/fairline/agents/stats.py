"""Season stats context via BALLDONTLIE, team-level for all four sports and
player-level for NFL/MLB (see PLAYER_STATS_SUPPORTED in the client module).

BALLDONTLIE's team IDs are internal to that API and unrelated to the
team-name strings on GameSnapshot, so every sport's fetch starts by
resolving names against a teams list fetched fresh each run (teams change
rarely enough that a per-run fetch, not a cache, is the simplest correct
choice). The free tier caps at 5 requests/minute; any fetch failure
(network error, rate limit, missing key) degrades to skipping that team's
stats rather than failing the run, matching the pattern in weather_agent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from fairline.clients.balldontlie import (
    PLAYER_STATS_SUPPORTED,
    SPORT_PATH,
    fetch_player_season_stats,
    fetch_team_season_stats,
    fetch_teams,
    resolve_team_id,
)
from fairline.state import FairlineState

logger = logging.getLogger(__name__)


def _current_season(sport: str, now: datetime | None = None) -> int:
    """The season year BALLDONTLIE expects. NFL/NBA/NHL seasons span two
    calendar years and are labeled by their start year, so a January game
    is still last year's season label. MLB seasons match the calendar year."""
    now = now or datetime.now(timezone.utc)
    if sport in {"americanfootball_nfl", "basketball_nba", "icehockey_nhl"} and now.month <= 6:
        return now.year - 1
    return now.year


async def stats_agent(state: FairlineState, client: httpx.AsyncClient) -> dict:
    """Attach team season stats, and NFL/MLB player season stats, per team in the slate."""
    games = state.get("games", [])
    sport = state.get("sport", "americanfootball_nfl")
    if not games or sport not in SPORT_PATH:
        return {"team_stats": {}, "player_stats": {}}

    bdl_sport = SPORT_PATH[sport]
    season = _current_season(sport)
    team_names = {g.home_team for g in games} | {g.away_team for g in games}

    try:
        teams = await fetch_teams(client, bdl_sport)
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.warning("stats_agent: teams lookup failed for %s: %s", sport, exc)
        return {"team_stats": {}, "player_stats": {}}

    team_stats: dict = {}
    player_stats: dict = {}
    for name in team_names:
        team_id = resolve_team_id(teams, name)
        if team_id is None:
            logger.warning("stats_agent: no BALLDONTLIE team match for %r (%s)", name, sport)
            continue

        try:
            stats = await fetch_team_season_stats(client, bdl_sport, team_id, season)
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("stats_agent: team stats fetch failed for %s: %s", name, exc)
            stats = None
        if stats:
            team_stats[name] = stats

        if bdl_sport in PLAYER_STATS_SUPPORTED:
            try:
                players = await fetch_player_season_stats(client, bdl_sport, team_id, season)
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning("stats_agent: player stats fetch failed for %s: %s", name, exc)
                players = []
            if players:
                player_stats[name] = players

    logger.info(
        "stats_agent: team stats for %d of %d teams, player stats for %d teams",
        len(team_stats), len(team_names), len(player_stats),
    )
    return {"team_stats": team_stats, "player_stats": player_stats}
