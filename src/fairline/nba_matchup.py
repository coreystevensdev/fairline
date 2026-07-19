"""NBA prop-matchup splits: last-5, last-10, season, home/away, and
back-to-back rest, on the same shrinkage-and-market-bound pattern already
proven for NFL, MLB, and NHL props.

Reuses matchup.py's sport-agnostic combine_splits rather than duplicating
the blending math.

Unlike MLB's vs_pitcher and NHL's vs_goalie, there is no vs-specific-defender
split here at all: no verified per-game defender-matchup data source exists
for NBA (the one endpoint that identifies who guarded whom,
LeagueSeasonMatchups, is season-aggregate only, confirmed during planning),
so this isn't a deferred feature, it's a genuine data gap.

No day/night or surface split exists for NBA; every game is played indoors
on a standardized court.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from fairline.db.models import NbaPlayerGame
from fairline.matchup import combine_splits

logger = logging.getLogger(__name__)

NBA_PROP_STAT_COLUMNS = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "three_pointers_made",
}


def compute_nba_prop_splits(
    games: list[NbaPlayerGame], stat: str, line: float,
    position_matchup: tuple[int, int] | None = None,
) -> dict[str, tuple[int, int]]:
    """Pre-registered NBA splits: last-N, season, home/away, back-to-back,
    plus an optional defense-vs-position split resolved by the caller (this
    function has no DB session, so it can't run the cross-player query
    itself -- position_matchup is computed by _opponent_position_rate and
    passed in)."""
    played = [g for g in games if getattr(g, stat) is not None]
    played.sort(key=lambda g: g.game_date, reverse=True)

    def rate(subset: list[NbaPlayerGame]) -> tuple[int, int]:
        hits = sum(1 for g in subset if getattr(g, stat) > line)
        return hits, len(subset)

    latest_season = played[0].season if played else 0
    splits = {
        "last_5": rate(played[:5]),
        "last_10": rate(played[:10]),
        "season": rate([g for g in played if g.season == latest_season]),
        "home": rate([g for g in played if g.is_home is True]),
        "away": rate([g for g in played if g.is_home is False]),
        "back_to_back": rate([g for g in played if g.rest_days is not None and g.rest_days <= 1]),
    }
    if position_matchup is not None:
        splits["position_matchup"] = position_matchup
    return splits


def nba_matchup_probability(
    games: list[NbaPlayerGame], stat: str, line: float, side: str, market_fair: float,
    position_matchup: tuple[int, int] | None = None,
) -> tuple[float, dict[str, tuple[int, int]]]:
    """Probability the side hits, with the splits that produced it."""
    splits = compute_nba_prop_splits(games, stat, line, position_matchup=position_matchup)
    over_fair = market_fair if side == "Over" else 1 - market_fair
    over_prob = combine_splits(splits, base_rate=over_fair, market_fair=over_fair)
    return (over_prob if side == "Over" else 1 - over_prob), splits


async def _player_position(session, player: str) -> str | None:
    """The player's own most-recently-recorded position, or None if unresolved."""
    row = (
        await session.execute(
            select(NbaPlayerGame.position)
            .where(NbaPlayerGame.player == player, NbaPlayerGame.position.is_not(None))
            .order_by(NbaPlayerGame.season.desc(), NbaPlayerGame.game_date.desc())
            .limit(1)
        )
    ).scalar()
    return row


async def _opponent_position_rate(
    session, opponent: str, position: str, stat: str, line: float
) -> tuple[int, int] | None:
    """How every player at `position` has fared against `opponent` on `stat`
    vs `line`, across all players who have faced them, not just one player's
    own history. Returns None when no qualifying games exist at all, so the
    caller can omit the split rather than showing a false (0, 0)."""
    column = getattr(NbaPlayerGame, stat)
    games = (
        (
            await session.execute(
                select(NbaPlayerGame).where(
                    NbaPlayerGame.opponent == opponent,
                    NbaPlayerGame.position == position,
                    column.is_not(None),
                )
            )
        )
        .scalars()
        .all()
    )
    if not games:
        return None
    hits = sum(1 for g in games if getattr(g, stat) > line)
    return hits, len(games)


def describe_nba_splits(splits: dict[str, tuple[int, int]], side: str, line: float) -> str:
    parts = [
        f"{name} {hits}-{attempts - hits} over {line:g}"
        for name, (hits, attempts) in splits.items()
        if attempts > 0
    ]
    return f"{side} angles: " + "; ".join(parts)


async def create_nba_matchup_candidates(session_factory, snapshot, min_edge: float = 0.03) -> int:
    """Queue NBA prop candidates where the splits-adjusted number beats
    retail, mirroring the MLB/NFL/NHL matchup-candidate creation flow."""
    import uuid

    from fairline.clients.odds_api import RETAIL_BOOKS
    from fairline.db.models import SteamCandidate
    from fairline.props import _paired_outcomes, prop_fair_lines
    from fairline.state import american_to_prob

    fair_by_key = {
        (fl.market, fl.player, fl.point): fl.over_prob for fl in prop_fair_lines(snapshot)
    }
    if not fair_by_key:
        return 0

    created = 0
    async with session_factory() as session:
        for bm in snapshot.bookmakers:
            if bm.key not in RETAIL_BOOKS:
                continue
            for (market, player, point), pair in _paired_outcomes(snapshot, bm.key).items():
                over_fair = fair_by_key.get((market, player, point))
                stat = NBA_PROP_STAT_COLUMNS.get(market)
                if over_fair is None or stat is None:
                    continue
                games = (
                    (
                        await session.execute(
                            select(NbaPlayerGame).where(NbaPlayerGame.player == player)
                        )
                    )
                    .scalars()
                    .all()
                )
                if not games:
                    continue
                own_team = games[0].team
                upcoming_opponent = (
                    snapshot.away_team if snapshot.home_team == own_team else snapshot.home_team
                )
                player_position = await _player_position(session, player)
                position_matchup = (
                    await _opponent_position_rate(session, upcoming_opponent, player_position, stat, point)
                    if player_position
                    else None
                )
                for side in ("Over", "Under"):
                    market_fair = over_fair if side == "Over" else 1 - over_fair
                    prob, splits = nba_matchup_probability(
                        games, stat, point, side, market_fair, position_matchup=position_matchup
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
                                + describe_nba_splits(splits, side, point)
                            ),
                            angles=",".join(
                                name for name, (_, attempts) in splits.items() if attempts > 0
                            ),
                            source="nba_matchup",
                            status="pending",
                        )
                    )
                    created += 1
        await session.commit()
    if created:
        logger.info("nba_matchup: %d prop candidates pending review", created)
    return created
