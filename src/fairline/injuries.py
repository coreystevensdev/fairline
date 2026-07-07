"""Injury context from ESPN's public feed (sim design Phase 3).

The design doc planned LLM extraction from injury reports; the feed turned
out to be structured JSON, so no LLM is involved at all. Code parses, code
applies bounded adjustments, and the notes reach the pick prompt so a
reviewer sees exactly which absence moved a number.

Magnitudes are market convention, not modeling: a starting NFL quarterback
is worth about five and a half points, everyone else far less, and the total
per team is capped because injury lists during bye weeks read like triage
wards without ever moving a real line that far.
"""

from __future__ import annotations

import logging

import httpx

from fairline.state import FairlineState

logger = logging.getLogger(__name__)

INJURY_URLS = {
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries",
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries",
}

_OUT_STATUSES = {"Out", "Doubtful", "Injured Reserve", "Suspension"}
_HALF_STATUSES = {"Questionable", "Day-To-Day"}

# Points of expected margin per absent starter, by sport and position group.
_POSITION_VALUE = {
    "americanfootball_nfl": {"QB": 5.5, "RB": 0.5, "WR": 0.5, "TE": 0.4, "default": 0.3},
    "basketball_nba": {"default": 1.0},
    "icehockey_nhl": {"G": 0.35, "default": 0.05},  # goals, applied to the lambda
}
_TEAM_CAP = 8.0


def parse_espn_injuries(payload: dict) -> dict[str, list[dict]]:
    """Team name -> [{player, position, status}], tolerant of feed gaps."""
    teams: dict[str, list[dict]] = {}
    for team in payload.get("injuries") or []:
        name = team.get("displayName")
        if not name:
            continue
        entries = []
        for item in team.get("injuries") or []:
            athlete = item.get("athlete") or {}
            player = athlete.get("displayName")
            status = item.get("status")
            if not player or not status:
                continue
            position = ((athlete.get("position") or {}).get("abbreviation")) or ""
            entries.append({"player": player, "position": position, "status": status})
        teams[name] = entries
    return teams


def injury_margin_adjustment(sport: str, entries: list[dict]) -> tuple[float, list[str]]:
    """Bounded negative adjustment for a team's absences, with the receipts."""
    values = _POSITION_VALUE.get(sport)
    if values is None:
        return 0.0, []
    total = 0.0
    notes = []
    for e in entries:
        if e["status"] in _OUT_STATUSES:
            factor = 1.0
        elif e["status"] in _HALF_STATUSES:
            factor = 0.5
        else:
            continue
        value = values.get(e["position"], values["default"])
        total += factor * value
        notes.append(f"{e['player']} ({e['position']}) {e['status']}")
    return -min(_TEAM_CAP, total), notes


async def injury_agent(state: FairlineState, client: httpx.AsyncClient) -> dict:
    """Attach injury adjustments for teams on today's slate."""
    games = state.get("games", [])
    sport = state.get("sport", "americanfootball_nfl")
    url = INJURY_URLS.get(sport)
    if not games or url is None:
        return {"team_injuries": {}}

    try:
        resp = await client.get(url, timeout=httpx.Timeout(15.0))
        resp.raise_for_status()
        by_team = parse_espn_injuries(resp.json())
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("injuries: feed failed for %s: %s", sport, exc)
        return {"team_injuries": {}}

    slate_teams = {t for g in games for t in (g.home_team, g.away_team)}
    result: dict = {}
    for team in slate_teams:
        adjustment, notes = injury_margin_adjustment(sport, by_team.get(team, []))
        if notes:
            result[team] = {"adjustment": adjustment, "notes": notes}
    logger.info("injury_agent: adjustments for %d of %d slate teams", len(result), len(slate_teams))
    return {"team_injuries": result}
