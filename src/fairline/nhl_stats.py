"""NHL skater game logs via the NHL's own official API.

No pitch-level aggregation is needed here, unlike MLB's Statcast pull: the
boxscore endpoint already reports each skater's per-game totals directly.
This module's job is walking a team's season schedule, deriving rest days
from consecutive game dates (the API has no rest-days field), and pulling
one boxscore per game to find the opposing starting goalie (the `starter`
flag) and each skater's stat line.
"""

from __future__ import annotations

import logging
from datetime import date

import httpx

from fairline.clients.nhl_api import fetch_boxscore, fetch_team_schedule
from fairline.db.models import NhlPlayerGame

logger = logging.getLogger(__name__)

_TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks", "SEA": "Seattle Kraken",
    "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Hockey Club", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}


def _derive_rest_days(games: list[dict]) -> list[int | None]:
    """Days since the previous game in this list, None for the first entry.
    Games must already be in chronological order."""
    rest: list[int | None] = [None]
    for prev, curr in zip(games, games[1:]):
        prev_date = date.fromisoformat(prev["game_date"])
        curr_date = date.fromisoformat(curr["game_date"])
        rest.append((curr_date - prev_date).days)
    return rest


def _opposing_starting_goalie(boxscore: dict, opponent_side: str) -> str | None:
    goalies = boxscore.get("playerByGameStats", {}).get(opponent_side, {}).get("goalies", [])
    starter = next((g for g in goalies if g.get("starter")), None)
    return starter["name"]["default"] if starter else None


async def fetch_nhl_skater_games(client: httpx.AsyncClient, team: str, season: str) -> list[NhlPlayerGame]:
    """Every skater's game log for one team's season, in NhlPlayerGame rows."""
    schedule = await fetch_team_schedule(client, team, season)
    schedule.sort(key=lambda g: g["game_date"])
    rest_days = _derive_rest_days(schedule)

    rows: list[NhlPlayerGame] = []
    for game, rest in zip(schedule, rest_days):
        is_home = game["home_team"] == team
        opponent_code = game["away_team"] if is_home else game["home_team"]
        own_side = "homeTeam" if is_home else "awayTeam"
        opponent_side = "awayTeam" if is_home else "homeTeam"

        boxscore = await fetch_boxscore(client, game["game_id"])
        opposing_goalie = _opposing_starting_goalie(boxscore, opponent_side)
        own_stats = boxscore.get("playerByGameStats", {}).get(own_side, {})
        skaters = (own_stats.get("forwards") or []) + (own_stats.get("defense") or [])

        for skater in skaters:
            rows.append(
                NhlPlayerGame(
                    season=int(season[:4]),
                    game_date=date.fromisoformat(game["game_date"]),
                    player=skater["name"]["default"],
                    team=_TEAM_NAMES.get(team, team),
                    opponent=_TEAM_NAMES.get(opponent_code, opponent_code),
                    opposing_goalie=opposing_goalie,
                    is_home=is_home,
                    rest_days=rest,
                    goals=skater.get("goals"),
                    assists=skater.get("assists"),
                    points=skater.get("points"),
                    shots_on_goal=skater.get("sog"),
                )
            )
    logger.info("nhl_stats: ingested %d skater-games for %s season %s", len(rows), team, season)
    return rows
