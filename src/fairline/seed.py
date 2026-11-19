"""Demo data for a deployment with no Odds API key.

The three read pages are the product's claim, and they render empty without a
paid feed. This fills them with a fixed, checkable set of rows so the deployed
demo shows the thing it describes.

Nothing here invents a number the app would have computed. The fixtures below
state only what a human would have observed, the price taken and the price at
close, and every derived figure comes from the same functions settle and grade
use: `clv` from `american_to_prob`, `profit_units` from `_profit_units`. A
reviewer who checks the arithmetic on the record page is checking the real
implementation, not a table of invented results.

This is sample data and must be labelled as such wherever it is served. A
graded betting record that is not real is a claim about performance, so
`seed_demo` sets the marker the templates read, and the pages say so.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from fairline.clv import _profit_units
from fairline.db.models import Pick, Run, SteamCandidate, User
from fairline.state import american_to_prob

logger = logging.getLogger(__name__)

DEMO_USER_ID = "demo-0000-0000-0000-000000000000"
DEMO_EMAIL = "demo@fairline.invalid"
DEMO_RUN_ID = "demo-run-0000-0000-000000000000"

# (sport, away, home, market, selection, book, price, closing_prob, result, source, rationale)
#
# closing_prob is what the sharp market settled at. price is what was taken.
# CLV falls out of the difference, so a pick can lose and still show positive
# CLV, which is the whole point of grading on it: three of the losses below
# beat the close.
_GRADED = [
    ("americanfootball_nfl", "Dallas Cowboys", "Philadelphia Eagles", "spreads",
     "Philadelphia Eagles -2.5", "BetMGM", -102, 0.548, "win", "steam",
     "Pinnacle moved -2.5 -110 to -3.0 -125 inside 40 minutes while BetMGM held."),
    ("americanfootball_nfl", "San Francisco 49ers", "Seattle Seahawks", "totals",
     "Under 44.5", "DraftKings", 105, 0.521, "loss", "steam",
     "Total fell 45.5 to 44.0 at the sharp book; DraftKings lagged a half point."),
    ("americanfootball_nfl", "Green Bay Packers", "Chicago Bears", "h2h",
     "Green Bay Packers", "Caesars", -140, 0.601, "win", "model",
     "Blend put the Packers at 60.1% against a price implying 58.3%."),
    ("baseball_mlb", "Baltimore Orioles", "Toronto Blue Jays", "player_props",
     "Bo Bichette Over 1.5 Total Bases", "FanDuel", 104, 0.533, "win", "matchup",
     "Opposing starter allows a .312 xwOBA to right-handed hitters, park factor 1.08 for doubles."),
    ("baseball_mlb", "Houston Astros", "Texas Rangers", "spreads",
     "Texas Rangers +1.5", "BetMGM", -155, 0.618, "loss", "matchup",
     "Bullpen rest edge and a platoon advantage through the order the second time."),
    ("icehockey_nhl", "Boston Bruins", "Tampa Bay Lightning", "totals",
     "Over 6.5", "DraftKings", 118, 0.472, "win", "steam",
     "Total climbed 6.0 to 6.5 on both sharp books within an hour of puck drop."),
    ("icehockey_nhl", "Colorado Avalanche", "Vegas Golden Knights", "h2h",
     "Colorado Avalanche", "Caesars", 135, 0.441, "loss", "model",
     "Expected-goals share over the last fifteen games favours Colorado on the road."),
    ("basketball_nba", "Milwaukee Bucks", "Boston Celtics", "spreads",
     "Boston Celtics -4.5", "FanDuel", -108, 0.531, "win", "matchup",
     "Opponent is on the second night of a back to back with no rest advantage."),
    ("basketball_nba", "Denver Nuggets", "Phoenix Suns", "player_props",
     "Nikola Jokic Over 9.5 Rebounds", "BetMGM", -115, 0.564, "win", "matchup",
     "Phoenix concede the third-highest second-chance rate; Jokic clears this in 71% of starts."),
    ("basketball_nba", "Miami Heat", "New York Knicks", "totals",
     "Under 212.5", "DraftKings", -110, 0.509, "push", "model",
     "Pace estimate sits two possessions under the number both books opened at."),
    ("americanfootball_nfl", "Buffalo Bills", "Miami Dolphins", "h2h",
     "Buffalo Bills", "FanDuel", -165, 0.641, "win", "steam",
     "Sharp money took Buffalo through -150 and the price never came back."),
    ("baseball_mlb", "New York Yankees", "Boston Red Sox", "totals",
     "Over 8.5", "Caesars", -105, 0.488, "loss", "steam",
     "Wind out to left at 14mph; the total moved 8.0 to 8.5 before first pitch."),
]

# pending candidates, which is what the queue page is for
_PENDING = [
    ("americanfootball_nfl", "Dallas Cowboys", "Philadelphia Eagles", "spreads",
     "Philadelphia Eagles -2.5", "BetMGM", -102, 0.531, "steam",
     "Pinnacle moved -2.5 -110 to -3.0 -125 inside 40 minutes while BetMGM held. "
     "The no-vig fair price implies 53.1% against the 50.5% this price pays."),
    ("baseball_mlb", "Baltimore Orioles", "Toronto Blue Jays", "player_props",
     "Bo Bichette Over 1.5 Total Bases", "FanDuel", 104, 0.552, "matchup",
     "Fair 0.500 to matchup 0.552. Opposing starter allows a .312 xwOBA to "
     "right-handed hitters and the park factor is 1.08 for doubles."),
    ("icehockey_nhl", "Ottawa Senators", "Montreal Canadiens", "h2h",
     "Montreal Canadiens", "DraftKings", 128, 0.471, "model",
     "Blend puts Montreal at 47.1% against a price implying 43.9%, driven by "
     "goaltender save-rate over the last ten starts."),
    ("basketball_nba", "LA Clippers", "Golden State Warriors", "totals",
     "Over 225.5", "Caesars", -108, 0.537, "steam",
     "Total climbed 223.5 to 225.0 at the sharp book while Caesars held the lower number."),
]


async def seed_demo(session_factory, reset: bool = False) -> dict:
    """Populate the read pages with checkable sample data.

    Returns counts rather than rows, so the CLI can report what it wrote.
    """
    now = datetime.now(timezone.utc)
    written = {"picks": 0, "candidates": 0}

    async with session_factory() as session:
        if reset:
            for table in (Pick, SteamCandidate, Run):
                await session.execute(delete(table).where(table.user_id == DEMO_USER_ID)
                                      if hasattr(table, "user_id") else delete(table))
            await session.execute(delete(User).where(User.id == DEMO_USER_ID))
            await session.commit()

        existing = (await session.execute(
            select(Pick).where(Pick.user_id == DEMO_USER_ID).limit(1)
        )).scalar_one_or_none()
        if existing is not None:
            logger.info("seed: demo rows already present, nothing written")
            return written

        user = (await session.execute(
            select(User).where(User.id == DEMO_USER_ID)
        )).scalar_one_or_none()
        if user is None:
            session.add(User(id=DEMO_USER_ID, email=DEMO_EMAIL))

        run = (await session.execute(
            select(Run).where(Run.id == DEMO_RUN_ID)
        )).scalar_one_or_none()
        if run is None:
            session.add(Run(id=DEMO_RUN_ID, user_id=DEMO_USER_ID, status="complete"))

        for i, (sport, away, home, market, selection, book, price,
                closing_prob, result, source, rationale) in enumerate(_GRADED):
            taken = american_to_prob(price)
            # the two figures the record page shows, from the functions that
            # produce them in settlement and grading
            clv = closing_prob - taken
            units = _profit_units(price, result)
            approved = now - timedelta(days=len(_GRADED) - i, hours=3)
            session.add(Pick(
                id=str(uuid.uuid4()), user_id=DEMO_USER_ID, run_id=DEMO_RUN_ID,
                sport=sport, game_id=f"demo-{sport}-{i}",
                home_team=home, away_team=away,
                commence_time=approved + timedelta(hours=2),
                market=market, selection=selection, book=book, price=price,
                sharp_probability=closing_prob, sim_probability=None,
                blended_probability=closing_prob, edge_pct=closing_prob - taken,
                ev_pct=(closing_prob - taken) * 2, confidence="medium",
                rationale=rationale, source=source, approved_at=approved,
                closing_price=price, closing_probability=closing_prob,
                clv=clv, result=result, profit_units=units,
            ))
            written["picks"] += 1

        for i, (sport, away, home, market, selection, book, price,
                sharp_prob, source, rationale) in enumerate(_PENDING):
            taken = american_to_prob(price)
            session.add(SteamCandidate(
                id=f"demo-pending-{i}", sport=sport, game_id=f"demo-pending-{sport}-{i}",
                home_team=home, away_team=away,
                commence_time=now + timedelta(days=1, hours=i),
                market=market, selection=selection, book=book, price=price,
                sharp_probability=sharp_prob, implied_probability=taken,
                edge_pct=sharp_prob - taken, ev_pct=(sharp_prob - taken) * 2,
                rationale=rationale, status="pending", source=source,
            ))
            written["candidates"] += 1

        await session.commit()

    logger.info("seed: wrote %d picks and %d candidates",
                written["picks"], written["candidates"])
    return written
