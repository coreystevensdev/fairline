"""Bulk NBA player-to-position lookup via nba_api's CommonTeamRoster.

Live verification (2026-07-16, real stats.nba.com call) found CommonTeamRoster's
POSITION column uses single-letter codes ("G", "F", "C") and letter combos
("G-F", "F-C"), not the "Guard"/"Forward"/"Center" words assumed during
planning. _bucket_position accepts both the letter codes and the full words,
taking the first listed position in a combo as the player's primary bucket.

Uses nba_api.stats.static.teams.get_teams() (a local, no-network list of all
30 real team IDs) plus one CommonTeamRoster call per team, 30 calls total
for the whole league -- not one call per player.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_BACKOFF_BASE_SECONDS = 2.0
_VALID_BUCKETS = {"Guard", "Forward", "Center"}
_LETTER_TO_BUCKET = {"G": "Guard", "F": "Forward", "C": "Center"}


def _bucket_position(position: str | None) -> str | None:
    """"G"/"Guard" -> "Guard"; "G-F"/"Guard-Forward" -> "Guard"; None/empty -> None."""
    if not position:
        return None
    first = position.split("-")[0].strip()
    if first in _VALID_BUCKETS:
        return first
    return _LETTER_TO_BUCKET.get(first)


def _get_teams() -> list[dict]:
    from nba_api.stats.static import teams

    return teams.get_teams()


def _call_team_roster(team_id: int, season: str, proxy: str | None, timeout: float):
    from nba_api.stats.endpoints import commonteamroster

    kwargs = {"team_id": team_id, "season": season, "timeout": timeout}
    if proxy:
        kwargs["proxy"] = proxy
    roster = commonteamroster.CommonTeamRoster(**kwargs)
    return roster.get_data_frames()[0]


def _fetch_sync(season: str, proxy: str | None, timeout: float, max_retries: int) -> dict[str, str]:
    positions: dict[str, str] = {}
    for team in _get_teams():
        team_id = team["id"]
        for attempt in range(max_retries):
            try:
                df = _call_team_roster(team_id, season, proxy, timeout)
                for row in df.to_dict(orient="records"):
                    bucket = _bucket_position(row.get("POSITION"))
                    if bucket:
                        positions[row["PLAYER"]] = bucket
                break
            except Exception as exc:  # nba_api/requests raise varied, undocumented exception types on a block
                if attempt < max_retries - 1:
                    sleep_for = _BACKOFF_BASE_SECONDS * (2 ** attempt)
                    logger.warning(
                        "nba_roster: team %s attempt %d/%d failed (%s), retrying in %.0fs",
                        team_id, attempt + 1, max_retries, exc, sleep_for,
                    )
                    time.sleep(sleep_for)
                else:
                    logger.warning("nba_roster: team %s failed after %d attempts, skipping", team_id, max_retries)
    return positions


async def fetch_league_positions(
    season: str, proxy: str | None = None, timeout: float = 60.0, max_retries: int = 3
) -> dict[str, str]:
    """{player_name: position_bucket} for every player in the league.

    One team's roster failing (after retries) is logged and skipped rather
    than aborting the whole league fetch, matching the graceful-degradation
    pattern used elsewhere in this codebase.
    """
    return await asyncio.to_thread(_fetch_sync, season, proxy, timeout, max_retries)
