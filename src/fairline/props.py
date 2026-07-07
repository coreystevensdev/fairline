"""Player prop devig and edge finding (props P1).

Same top-down method as game lines, applied per player: devig the sharp
book's Over/Under pair into a fair probability, then flag retail books
posting the same line at a worse-for-them price. Only exact point matches
compare; a retail 280.5 against a sharp 275.5 is a different bet, and
pricing the half-yards between lines needs a projection model this phase
does not have.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from fairline.agents.odds import best_sharp_book
from fairline.clients.odds_api import RETAIL_BOOKS
from fairline.state import GameSnapshot, american_to_prob, remove_vig

logger = logging.getLogger(__name__)


class PropLine(NamedTuple):
    market: str
    player: str
    point: float
    over_prob: float  # no-vig
    book: str
    over_price: int
    under_price: int


class PropEdge(NamedTuple):
    market: str
    player: str
    point: float
    side: str
    book: str
    price: int
    fair_prob: float
    implied_prob: float
    edge_pct: float


def _paired_outcomes(snapshot: GameSnapshot, book_key: str) -> dict:
    """(market, player, point) -> {"Over": Outcome, "Under": Outcome} for one book."""
    bm = next((b for b in snapshot.bookmakers if b.key == book_key), None)
    if bm is None:
        return {}
    pairs: dict = {}
    for mkt in bm.markets:
        for o in mkt.outcomes:
            if o.description is None or o.point is None:
                continue
            pairs.setdefault((mkt.key, o.description, o.point), {})[o.name] = o
    return {k: v for k, v in pairs.items() if "Over" in v and "Under" in v}


def prop_fair_lines(snapshot: GameSnapshot) -> list[PropLine]:
    """No-vig fair over probability per player line at the sharp book."""
    book = best_sharp_book(snapshot)
    if book is None:
        return []
    lines = []
    for (market, player, point), pair in _paired_outcomes(snapshot, book).items():
        fair = remove_vig([american_to_prob(pair["Over"].price), american_to_prob(pair["Under"].price)])
        lines.append(
            PropLine(
                market=market,
                player=player,
                point=point,
                over_prob=fair[0],
                book=book,
                over_price=pair["Over"].price,
                under_price=pair["Under"].price,
            )
        )
    return lines


def find_prop_edges(snapshot: GameSnapshot, min_edge: float = 0.03) -> list[PropEdge]:
    """Retail prop prices beating the sharp fair number by min_edge or more."""
    fair_by_key = {
        (fl.market, fl.player, fl.point): fl.over_prob for fl in prop_fair_lines(snapshot)
    }
    if not fair_by_key:
        return []

    edges = []
    for bm in snapshot.bookmakers:
        if bm.key not in RETAIL_BOOKS:
            continue
        for (market, player, point), pair in _paired_outcomes(snapshot, bm.key).items():
            over_prob = fair_by_key.get((market, player, point))
            if over_prob is None:
                continue
            for side, fair_prob in (("Over", over_prob), ("Under", 1 - over_prob)):
                implied = american_to_prob(pair[side].price)
                edge = fair_prob - implied
                if edge >= min_edge:
                    edges.append(
                        PropEdge(
                            market=market,
                            player=player,
                            point=point,
                            side=side,
                            book=bm.key,
                            price=pair[side].price,
                            fair_prob=fair_prob,
                            implied_prob=implied,
                            edge_pct=edge,
                        )
                    )
    edges.sort(key=lambda e: -e.edge_pct)
    return edges


async def settle_prop_picks(
    snapshots: list[GameSnapshot],
    session_factory,
    now=None,
    window_minutes: int = 30,
) -> dict:
    """Capture closing prop lines for unsettled prop picks near kickoff.

    Prop lines drift far more than game spreads, so CLV is computed only when
    the sharp book's closing point equals the point the pick was taken at.
    Drift is still recorded: closing_point and closing_price land either way,
    and a NULL clv next to a moved point is the honest reading.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select

    from fairline.db.models import Pick
    from fairline.matchup import PROP_STAT_COLUMNS, _parse_prop_selection

    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now + timedelta(minutes=window_minutes)
    lines_by_game: dict[str, list[PropLine]] = {
        snap.game_id: prop_fair_lines(snap) for snap in snapshots
    }

    settled = point_moved = missed = pending = 0
    async with session_factory() as session:
        picks = (
            (
                await session.execute(
                    select(Pick).where(
                        Pick.closing_price.is_(None), Pick.market.in_(PROP_STAT_COLUMNS)
                    )
                )
            )
            .scalars()
            .all()
        )
        for pick in picks:
            commence = pick.commence_time
            if commence is not None and commence.tzinfo is None:
                commence = commence.replace(tzinfo=timezone.utc)
            if commence is None or commence > cutoff:
                pending += 1
                continue
            parsed = _parse_prop_selection(pick.selection)
            if parsed is None:
                missed += 1
                continue
            player, side, taken_point = parsed
            candidates = [
                fl
                for fl in lines_by_game.get(pick.game_id, [])
                if fl.market == pick.market and fl.player == player
            ]
            if not candidates:
                missed += 1
                logger.warning(
                    "prop settle: no closing line for pick_id=%s %r", pick.id, pick.selection
                )
                continue
            line = min(candidates, key=lambda fl: abs(fl.point - taken_point))
            side_prob = line.over_prob if side == "Over" else 1 - line.over_prob
            pick.closing_point = line.point
            pick.closing_price = line.over_price if side == "Over" else line.under_price
            pick.closing_probability = side_prob
            if line.point == taken_point:
                pick.clv = side_prob - american_to_prob(pick.price)
                settled += 1
            else:
                point_moved += 1
        await session.commit()

    return {
        "settled": settled,
        "point_moved": point_moved,
        "missed": missed,
        "pending": pending,
    }
