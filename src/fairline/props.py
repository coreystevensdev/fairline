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
        lines.append(PropLine(market=market, player=player, point=point, over_prob=fair[0], book=book))
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
