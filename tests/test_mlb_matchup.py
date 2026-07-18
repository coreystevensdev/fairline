"""Tests for MLB splits computation: new dimensions, sample-size floor on
vs-pitcher, reuse of matchup.py's sport-agnostic shrinkage math."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fairline.db.models import MlbPlayerGame
from fairline.mlb_matchup import (
    MIN_VS_PITCHER_SAMPLE,
    MLB_PROP_STAT_COLUMNS,
    compute_mlb_prop_splits,
    describe_mlb_splits,
    mlb_matchup_probability,
)


def _game(hits=1, day_night="night", is_home=True, opponent="Boston Red Sox",
          opposing_pitcher="Brayan Bello", team="New York Yankees", season=2025, date=None):
    return MlbPlayerGame(
        season=season,
        game_date=date or datetime(2025, 6, 14, tzinfo=timezone.utc),
        player="Aaron Judge",
        team=team,
        opponent=opponent,
        opposing_pitcher=opposing_pitcher,
        is_home=is_home,
        day_night=day_night,
        at_bats=4,
        hits=hits,
        home_runs=0,
        rbis=0,
        total_bases=hits,
        strikeouts=0,
        walks=0,
    )


class TestComputeMlbPropSplits:
    def test_day_night_home_away_splits_present(self):
        games = [_game(day_night="night", is_home=True), _game(day_night="day", is_home=False, hits=0)]
        splits = compute_mlb_prop_splits(games, "hits", 0.5)
        assert splits["night"] == (1, 1)
        assert splits["day"] == (0, 1)
        assert splits["home"] == (1, 1)
        assert splits["away"] == (0, 1)

    def test_vs_pitcher_below_floor_is_withheld(self):
        games = [_game(hits=1)] * 3  # only 3 PA vs this pitcher, below MIN_VS_PITCHER_SAMPLE
        splits = compute_mlb_prop_splits(games, "hits", 0.5, opposing_pitcher="Brayan Bello")
        assert "vs_pitcher" not in splits

    def test_vs_pitcher_at_or_above_floor_is_included(self):
        games = [_game(hits=1)] * MIN_VS_PITCHER_SAMPLE
        splits = compute_mlb_prop_splits(games, "hits", 0.5, opposing_pitcher="Brayan Bello")
        assert splits["vs_pitcher"] == (MIN_VS_PITCHER_SAMPLE, MIN_VS_PITCHER_SAMPLE)

    def test_park_factor_split_filters_to_matching_bucket(self):
        games = [
            _game(team="Colorado Rockies", opponent="San Diego Padres", is_home=True, hits=1),
            _game(team="Atlanta Braves", opponent="San Diego Padres", is_home=True, hits=0),
        ]
        splits = compute_mlb_prop_splits(games, "hits", 0.5, upcoming_park_bucket="hitter_park")
        assert splits["park_factor"] == (1, 1)  # only the Rockies (hitter-park) game counts

    def test_neutral_upcoming_park_omits_the_split(self):
        games = [_game(team="Atlanta Braves", is_home=True)]
        splits = compute_mlb_prop_splits(games, "hits", 0.5, upcoming_park_bucket=None)
        assert "park_factor" not in splits


def test_mlb_prop_stat_columns_covers_five_batter_markets():
    assert MLB_PROP_STAT_COLUMNS == {
        "batter_hits": "hits",
        "batter_home_runs": "home_runs",
        "batter_rbis": "rbis",
        "batter_total_bases": "total_bases",
        "batter_strikeouts": "strikeouts",
    }


def test_mlb_matchup_probability_bounded_near_market():
    games = [_game(hits=1)] * 10
    prob, splits = mlb_matchup_probability(games, "hits", 0.5, "Over", market_fair=0.55)
    # 0.55 + 0.06 clamp boundary; +1e-9 absorbs the float rounding of that add
    assert 0.49 <= prob <= 0.61 + 1e-9


def test_describe_mlb_splits_lists_only_present_splits():
    splits = {"night": (3, 5), "day": (0, 0)}
    text = describe_mlb_splits(splits, "Over", 0.5)
    assert "night 3-2 over 0.5" in text
    assert "day" not in text  # zero-attempt splits are excluded, same as NFL's describe_splits
