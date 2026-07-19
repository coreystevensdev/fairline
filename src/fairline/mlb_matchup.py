"""MLB prop-matchup splits: day/night, home/away, park factor, and
vs-specific-pitcher, layered on the last-5/last-10/season pattern already
proven for NFL props.

Reuses matchup.py's sport-agnostic shrinkage math (shrunk_probability,
combine_splits) rather than duplicating it -- that part of the pattern
doesn't change per sport, only which splits get computed does.

Vs-pitcher is the one split with a real small-sample risk (a batter can
face a specific starter as few as 0-5 times in a season), so it is the
only split with a hard floor: below MIN_VS_PITCHER_SAMPLE attempts it is
omitted from the returned dict entirely, not shown with shrinkage alone.
Day/night, home/away, and park-factor splits get shrinkage-only treatment
like the existing NFL splits, since a full season gives those real sample
size.

Vs-pitcher is now wired into create_mlb_matchup_candidates: the probable
starting pitcher for an upcoming game comes from MLB's own free official
Stats API (see clients/mlb_schedule_client.py), resolved per-batter by
comparing their most-recently-recorded team against both sides of the
snapshot. If that team matches neither side, or the schedule has no
probable pitcher listed yet for the matching side, the split is omitted
rather than guessed -- the same graceful-degradation principle used for the
NBA and NFL defense-vs-position splits elsewhere in this codebase.
"""

from __future__ import annotations

import logging

import httpx
from sqlalchemy import select

from fairline.clients.mlb_schedule_client import fetch_probable_pitchers, resolve_probable_pitcher
from fairline.db.models import MlbPlayerGame
from fairline.matchup import combine_splits
from fairline.mlb_park_factors import game_park_bucket

logger = logging.getLogger(__name__)

MLB_PROP_STAT_COLUMNS = {
    "batter_hits": "hits",
    "batter_home_runs": "home_runs",
    "batter_rbis": "rbis",
    "batter_total_bases": "total_bases",
    "batter_strikeouts": "strikeouts",
}

MIN_VS_PITCHER_SAMPLE = 10


def compute_mlb_prop_splits(
    games: list[MlbPlayerGame],
    stat: str,
    line: float,
    opposing_pitcher: str | None = None,
    upcoming_park_bucket: str | None = None,
) -> dict[str, tuple[int, int]]:
    """Pre-registered MLB splits: last-N, season, day/night, home/away, park,
    and (when a real sample exists) vs the specific opposing pitcher."""
    played = [g for g in games if getattr(g, stat) is not None]
    played.sort(key=lambda g: g.game_date, reverse=True)

    def rate(subset: list[MlbPlayerGame]) -> tuple[int, int]:
        hits = sum(1 for g in subset if getattr(g, stat) > line)
        return hits, len(subset)

    latest_season = played[0].season if played else 0
    splits: dict[str, tuple[int, int]] = {
        "last_5": rate(played[:5]),
        "last_10": rate(played[:10]),
        "season": rate([g for g in played if g.season == latest_season]),
        "day": rate([g for g in played if g.day_night == "day"]),
        "night": rate([g for g in played if g.day_night == "night"]),
        "home": rate([g for g in played if g.is_home]),
        "away": rate([g for g in played if not g.is_home]),
    }

    if opposing_pitcher:
        vs_pitcher = rate([g for g in played if g.opposing_pitcher == opposing_pitcher])
        if vs_pitcher[1] >= MIN_VS_PITCHER_SAMPLE:
            splits["vs_pitcher"] = vs_pitcher
        # below the floor: a batter's handful of at-bats against one pitcher
        # is close to statistically meaningless, so it's withheld outright

    if upcoming_park_bucket:
        park_games = [g for g in played if game_park_bucket(g) == upcoming_park_bucket]
        splits["park_factor"] = rate(park_games)

    return splits


def _player_current_team(games: list[MlbPlayerGame]) -> str:
    """The team from the most recent game in an already-fetched games list --
    a fetch with no ORDER BY returns rows in an unspecified order, so trusting
    games[0] risks a stale team for a batter who changed teams mid-season."""
    latest = max(games, key=lambda g: g.game_date)
    return latest.team


def mlb_matchup_probability(
    games: list[MlbPlayerGame],
    stat: str,
    line: float,
    side: str,
    market_fair: float,
    opposing_pitcher: str | None = None,
    upcoming_park_bucket: str | None = None,
) -> tuple[float, dict[str, tuple[int, int]]]:
    """Probability the side hits, with the splits that produced it."""
    splits = compute_mlb_prop_splits(games, stat, line, opposing_pitcher, upcoming_park_bucket)
    over_fair = market_fair if side == "Over" else 1 - market_fair
    over_prob = combine_splits(splits, base_rate=over_fair, market_fair=over_fair)
    return (over_prob if side == "Over" else 1 - over_prob), splits


def describe_mlb_splits(splits: dict[str, tuple[int, int]], side: str, line: float) -> str:
    parts = [
        f"{name} {hits}-{attempts - hits} over {line:g}"
        for name, (hits, attempts) in splits.items()
        if attempts > 0
    ]
    return f"{side} angles: " + "; ".join(parts)


async def create_mlb_matchup_candidates(session_factory, snapshot, min_edge: float = 0.03) -> int:
    """Queue MLB batter-prop candidates where the splits-adjusted number beats
    retail, mirroring matchup.py's create_matchup_candidates for NFL props.

    opposing_pitcher is resolved per-batter from MLB's probable-pitcher
    schedule (see module docstring); park_factor uses the upcoming game's
    actual host team.
    """
    import uuid

    from fairline.clients.odds_api import RETAIL_BOOKS
    from fairline.db.models import SteamCandidate
    from fairline.mlb_park_factors import park_bucket
    from fairline.props import _paired_outcomes, prop_fair_lines
    from fairline.state import american_to_prob

    fair_by_key = {
        (fl.market, fl.player, fl.point): fl.over_prob for fl in prop_fair_lines(snapshot)
    }
    if not fair_by_key:
        return 0

    upcoming_park_bucket = park_bucket(snapshot.home_team)
    async with httpx.AsyncClient() as http_client:
        probable_games = await fetch_probable_pitchers(
            http_client, snapshot.commence_time.date().isoformat()
        )
    created = 0
    async with session_factory() as session:
        for bm in snapshot.bookmakers:
            if bm.key not in RETAIL_BOOKS:
                continue
            for (market, player, point), pair in _paired_outcomes(snapshot, bm.key).items():
                over_fair = fair_by_key.get((market, player, point))
                stat = MLB_PROP_STAT_COLUMNS.get(market)
                if over_fair is None or stat is None:
                    continue
                games = (
                    (
                        await session.execute(
                            select(MlbPlayerGame).where(MlbPlayerGame.player == player)
                        )
                    )
                    .scalars()
                    .all()
                )
                if not games:
                    continue
                own_team = _player_current_team(games)
                # If the batter's resolved team matches neither side of the snapshot
                # (a name-normalization drift, or a prop feed lagging a trade), the
                # opposing pitcher can't be derived safely -- omit the split rather
                # than guessing.
                if own_team not in (snapshot.home_team, snapshot.away_team):
                    opposing_pitcher = None
                else:
                    opposing_side = "away" if own_team == snapshot.home_team else "home"
                    opposing_pitcher = resolve_probable_pitcher(
                        probable_games, snapshot.home_team, snapshot.away_team,
                        snapshot.commence_time, opposing_side,
                    )
                for side in ("Over", "Under"):
                    market_fair = over_fair if side == "Over" else 1 - over_fair
                    prob, splits = mlb_matchup_probability(
                        games, stat, point, side, market_fair,
                        opposing_pitcher=opposing_pitcher,
                        upcoming_park_bucket=upcoming_park_bucket,
                    )
                    implied = american_to_prob(pair[side].price)
                    edge = prob - implied
                    if edge < min_edge:
                        continue
                    selection = f"{player} {side} {point:g}"
                    already = (
                        await session.execute(
                            select(SteamCandidate.id).where(
                                SteamCandidate.game_id == snapshot.game_id,
                                SteamCandidate.market == market,
                                SteamCandidate.selection == selection,
                                SteamCandidate.book == bm.key,
                                SteamCandidate.status == "pending",
                            )
                        )
                    ).scalar()
                    if already:
                        continue
                    price = pair[side].price
                    win_amount = price / 100 if price > 0 else 100 / abs(price)
                    session.add(
                        SteamCandidate(
                            id=str(uuid.uuid4()),
                            sport=snapshot.sport,
                            game_id=snapshot.game_id,
                            home_team=snapshot.home_team,
                            away_team=snapshot.away_team,
                            commence_time=snapshot.commence_time,
                            market=market,
                            selection=selection,
                            book=bm.key,
                            price=price,
                            sharp_probability=prob,
                            implied_probability=implied,
                            edge_pct=edge,
                            ev_pct=prob * win_amount - (1 - prob),
                            rationale=(
                                f"fair {market_fair:.3f} -> matchup {prob:.3f}; "
                                + describe_mlb_splits(splits, side, point)
                            ),
                            angles=",".join(
                                name for name, (_, attempts) in splits.items() if attempts > 0
                            ),
                            source="mlb_matchup",
                            status="pending",
                        )
                    )
                    created += 1
        await session.commit()
    if created:
        logger.info("mlb_matchup: %d prop candidates pending review", created)
    return created
