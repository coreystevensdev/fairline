"""Static MLB park-factor table and hitter/pitcher-park bucketing.

Source: FanGraphs Guts tool (fangraphs.com/guts.aspx?type=pf), 2025 season,
"Basic (5yr)" overall park factor column (100 = league average; a 5-year
regressed figure, not single-season noise). Fetched and verified 2026-07-18.
Park factors barely move within a season, so a static table refreshed once
a year is the right tool here, not a live scrape against a page with no
documented API (Baseball Savant's own park-factor leaderboard is
JS-rendered and returned no usable data via direct fetch when this table
was built).
"""

from __future__ import annotations

from fairline.db.models import MlbPlayerGame

MLB_PARK_FACTORS: dict[str, int] = {
    "Los Angeles Angels": 101,
    "Baltimore Orioles": 99,
    "Boston Red Sox": 104,
    "Chicago White Sox": 100,
    "Cleveland Guardians": 99,
    "Detroit Tigers": 100,
    "Kansas City Royals": 103,
    "Minnesota Twins": 101,
    "New York Yankees": 99,
    "Athletics": 103,
    "Seattle Mariners": 94,
    "Tampa Bay Rays": 101,
    "Texas Rangers": 99,
    "Toronto Blue Jays": 99,
    "Arizona Diamondbacks": 101,
    "Atlanta Braves": 100,
    "Chicago Cubs": 98,
    "Cincinnati Reds": 105,
    "Colorado Rockies": 113,
    "Miami Marlins": 101,
    "Houston Astros": 99,
    "Los Angeles Dodgers": 99,
    "Milwaukee Brewers": 99,
    "Washington Nationals": 100,
    "New York Mets": 96,
    "Philadelphia Phillies": 101,
    "Pittsburgh Pirates": 102,
    "St. Louis Cardinals": 98,
    "San Diego Padres": 96,
    "San Francisco Giants": 97,
}

# Thresholds are a documented heuristic, not a statistical cutoff: anything
# clearly above/below the pack gets bucketed, everything else (most parks)
# reads as neutral and contributes no park-factor split at all.
_HITTER_THRESHOLD = 103
_PITCHER_THRESHOLD = 97


def park_bucket(team: str) -> str | None:
    """Hitter-friendly, pitcher-friendly, or None for neutral/unrecognized."""
    factor = MLB_PARK_FACTORS.get(team)
    if factor is None:
        return None
    if factor > _HITTER_THRESHOLD:
        return "hitter_park"
    if factor < _PITCHER_THRESHOLD:
        return "pitcher_park"
    return None


def game_park_bucket(game: MlbPlayerGame) -> str | None:
    """Bucket for the park a historical game was actually played in.

    The park always belongs to whichever team hosted, not the player's own
    team when they were the visitor.
    """
    host = game.team if game.is_home else game.opponent
    return park_bucket(host)
