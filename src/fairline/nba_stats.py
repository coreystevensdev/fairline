"""NBA player game logs via nba_api's LeagueGameLog, parsed into per-game rows.

Home/away and opponent aren't separate fields on this endpoint, they're
encoded in the MATCHUP string ("LAL vs. BOS" for a home game, "LAL @ BOS"
for an away game). Rest days aren't on the endpoint at all and are derived
from consecutive GAME_DATE values per player, the same shape used for NHL.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from fairline.clients.nba_stats_client import fetch_league_game_log
from fairline.db.models import NbaPlayerGame

logger = logging.getLogger(__name__)

_TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


def _parse_matchup(matchup: str) -> tuple[bool, str]:
    """"LAL vs. BOS" -> (True, "BOS"); "LAL @ BOS" -> (False, "BOS")."""
    if " vs. " in matchup:
        _, opponent_code = matchup.split(" vs. ")
        return True, opponent_code.strip()
    _, opponent_code = matchup.split(" @ ")
    return False, opponent_code.strip()


def _derive_rest_days(games: list[dict]) -> list[int | None]:
    """Days since the previous game in this list, None for the first entry.
    Games must already be in chronological order."""
    rest: list[int | None] = [None]
    for prev, curr in zip(games, games[1:]):
        prev_date = date.fromisoformat(prev["GAME_DATE"])
        curr_date = date.fromisoformat(curr["GAME_DATE"])
        rest.append((curr_date - prev_date).days)
    return rest


async def fetch_nba_player_games(season: str, proxy: str | None = None) -> list[NbaPlayerGame]:
    """Every player's game log for one NBA season, in NbaPlayerGame rows."""
    rows = await fetch_league_game_log(season, proxy=proxy)

    by_player: dict[str, list[dict]] = {}
    for row in rows:
        by_player.setdefault(row["PLAYER_NAME"], []).append(row)

    result: list[NbaPlayerGame] = []
    for player, games in by_player.items():
        games.sort(key=lambda g: g["GAME_DATE"])
        rest_days = _derive_rest_days(games)
        for game, rest in zip(games, rest_days):
            is_home, opponent_code = _parse_matchup(game["MATCHUP"])
            team_code = game["TEAM_ABBREVIATION"]
            result.append(
                NbaPlayerGame(
                    season=int(season[:4]),
                    game_date=datetime.combine(
                        date.fromisoformat(game["GAME_DATE"]), datetime.min.time(), tzinfo=timezone.utc
                    ),
                    player=player,
                    team=_TEAM_NAMES.get(team_code, team_code),
                    opponent=_TEAM_NAMES.get(opponent_code, opponent_code),
                    is_home=is_home,
                    rest_days=rest,
                    points=game.get("PTS"),
                    rebounds=game.get("REB"),
                    assists=game.get("AST"),
                    three_pointers_made=game.get("FG3M"),
                )
            )
    logger.info("nba_stats: ingested %d player-games for season %s", len(result), season)
    return result
