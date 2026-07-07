"""Tests for prop odds ingest and devig: fair lines and retail edges."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from fairline.clients.odds_api import fetch_event_props
from fairline.props import find_prop_edges, prop_fair_lines
from fairline.state import BookmakerOdds, GameSnapshot, MarketOdds, Outcome, american_to_prob

KICKOFF = datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc)
_ODDS_BASE = "https://api.the-odds-api.com/v4"


def _prop_outcome(side: str, price: int, player: str = "Patrick Mahomes", point: float = 275.5) -> Outcome:
    return Outcome(name=side, price=price, point=point, description=player)


def _snapshot(retail_over=105, retail_under=-125, retail_point=275.5) -> GameSnapshot:
    return GameSnapshot(
        game_id="evt-1",
        sport="americanfootball_nfl",
        home_team="Kansas City Chiefs",
        away_team="Las Vegas Raiders",
        commence_time=KICKOFF,
        bookmakers=[
            BookmakerOdds(
                key="pinnacle",
                title="Pinnacle",
                markets=[
                    MarketOdds(
                        key="player_pass_yds",
                        outcomes=[
                            _prop_outcome("Over", -115),
                            _prop_outcome("Under", -105),
                        ],
                    )
                ],
            ),
            BookmakerOdds(
                key="draftkings",
                title="DraftKings",
                markets=[
                    MarketOdds(
                        key="player_pass_yds",
                        outcomes=[
                            _prop_outcome("Over", retail_over, point=retail_point),
                            _prop_outcome("Under", retail_under, point=retail_point),
                        ],
                    )
                ],
            ),
        ],
    )


def test_prop_fair_lines_devigs_the_sharp_pair():
    lines = prop_fair_lines(_snapshot())

    assert len(lines) == 1
    line = lines[0]
    assert line.player == "Patrick Mahomes"
    assert line.point == 275.5
    over_raw = american_to_prob(-115)
    under_raw = american_to_prob(-105)
    assert line.over_prob == pytest.approx(over_raw / (over_raw + under_raw))


def test_find_prop_edges_flags_stale_retail_over():
    # sharp fair over ~.511; DK Over +105 implies .488 -> ~2.3 points of edge
    edges = find_prop_edges(_snapshot(), min_edge=0.02)

    assert len(edges) == 1
    e = edges[0]
    assert e.side == "Over"
    assert e.book == "draftkings"
    assert e.price == 105
    assert e.edge_pct == pytest.approx(0.0230, abs=0.002)


def test_find_prop_edges_skips_mismatched_points():
    edges = find_prop_edges(_snapshot(retail_point=280.5), min_edge=0.0)
    assert edges == []


def test_find_prop_edges_reads_the_under_side_too():
    # cheap retail Under vs fair under prob ~.489
    edges = find_prop_edges(_snapshot(retail_over=-200, retail_under=125), min_edge=0.02)

    assert len(edges) == 1
    assert edges[0].side == "Under"


def test_no_sharp_book_means_no_lines():
    snap = _snapshot()
    snap = snap.model_copy(update={"bookmakers": snap.bookmakers[1:]})  # drop pinnacle
    assert prop_fair_lines(snap) == []
    assert find_prop_edges(snap) == []


@pytest.mark.asyncio
@respx.mock
async def test_fetch_event_props_hits_event_endpoint(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    event = {
        "id": "evt-1",
        "sport_key": "americanfootball_nfl",
        "commence_time": "2026-01-15T20:00:00Z",
        "home_team": "Kansas City Chiefs",
        "away_team": "Las Vegas Raiders",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [
                    {
                        "key": "player_pass_yds",
                        "outcomes": [
                            {"name": "Over", "price": -115, "point": 275.5, "description": "Patrick Mahomes"},
                            {"name": "Under", "price": -105, "point": 275.5, "description": "Patrick Mahomes"},
                        ],
                    }
                ],
            }
        ],
    }
    route = respx.get(f"{_ODDS_BASE}/sports/americanfootball_nfl/events/evt-1/odds").mock(
        return_value=httpx.Response(200, json=event)
    )
    async with httpx.AsyncClient() as client:
        snap = await fetch_event_props(client, "americanfootball_nfl", "evt-1")

    assert route.called
    assert snap is not None
    outcome = snap.bookmakers[0].markets[0].outcomes[0]
    assert outcome.description == "Patrick Mahomes"
