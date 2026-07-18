"""Park-factor lookup and bucketing: static table, no live fetch."""

from __future__ import annotations

from datetime import datetime, timezone

from fairline.db.models import MlbPlayerGame
from fairline.mlb_park_factors import MLB_PARK_FACTORS, game_park_bucket, park_bucket


def test_all_30_teams_present():
    assert len(MLB_PARK_FACTORS) == 30


def test_rockies_is_a_hitter_park():
    assert park_bucket("Colorado Rockies") == "hitter_park"


def test_mariners_is_a_pitcher_park():
    assert park_bucket("Seattle Mariners") == "pitcher_park"


def test_braves_is_neutral():
    assert park_bucket("Atlanta Braves") is None


def test_unknown_team_returns_none():
    assert park_bucket("Montreal Expos") is None


def _game(team="Colorado Rockies", opponent="San Diego Padres", is_home=True) -> MlbPlayerGame:
    return MlbPlayerGame(
        season=2025,
        game_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
        player="Test Player",
        team=team,
        opponent=opponent,
        is_home=is_home,
        day_night="day",
    )


def test_game_park_bucket_uses_host_team_when_player_team_is_home():
    assert game_park_bucket(_game(team="Colorado Rockies", is_home=True)) == "hitter_park"


def test_game_park_bucket_uses_opponent_when_player_team_is_away():
    # player's team (Padres) is away; the actual host is the Rockies, a hitter park
    assert game_park_bucket(_game(team="San Diego Padres", opponent="Colorado Rockies", is_home=False)) == "hitter_park"
