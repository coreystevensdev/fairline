"""Server-rendered pages: leaderboard, review queue, the record.

Read pages are public on purpose: the graded record is the product's claim,
and a claim behind a login is marketing. Mutations (approve/reject) go to the
existing authed API endpoints; the page attaches the operator's key from
localStorage via htmx.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select

from fairline.db.session import get_session_factory
from fairline.seed import DEMO_USER_ID

_WEB = Path(__file__).resolve().parent.parent / "web"
templates = Jinja2Templates(directory=str(_WEB / "templates"))
router = APIRouter()


async def _is_demo_data(factory) -> bool:
    """Whether these pages are serving rows from `fairline seed-demo`.

    A graded record is this product's claim, so a seeded one has to say so on
    every page that shows it. The env var is an operator's sticky override: a
    deployment that seeds and then takes real picks keeps saying so until
    someone unsets it. The query is the floor under that, because the only
    dangerous direction is a seeded record served with no label, and a flag
    nobody remembered to set defaults to exactly that.
    """
    if os.getenv("FAIRLINE_DEMO_DATA", "").lower() == "true":
        return True
    if factory is None:
        return False

    from fairline.db.models import Pick

    async with factory() as session:
        seeded = await session.execute(
            select(Pick.id).where(Pick.user_id == DEMO_USER_ID).limit(1)
        )
        return seeded.scalar_one_or_none() is not None


def _factory():
    try:
        return get_session_factory()
    except RuntimeError:
        return None


# The public pages show one operator's record, not everyone's. The schema and
# the authed API are already multi-tenant (routes.py filters Pick.user_id ==
# caller), so the unscoped page queries were the odd ones out. With Stripe
# subscriptions in this repo, the second paying user's selections, prices and
# results would have been readable by any anonymous visitor at /record. Only the
# seeded demo user has rows today, which is what kept it latent rather than
# visible, and latent is not a reason to leave it.
#
# A setting rather than a literal so a real operator can publish their own
# record instead of the demo's.
PUBLIC_RECORD_USER_ID = os.environ.get("FAIRLINE_PUBLIC_RECORD_USER_ID", DEMO_USER_ID)


@router.get("/", include_in_schema=False)
async def agents_page(request: Request):
    from fairline.clv import agent_report

    factory = _factory()
    report = await agent_report(factory, user_id=PUBLIC_RECORD_USER_ID) if factory else {}
    return templates.TemplateResponse(
        request, "agents.html", {"page": "agents", "report": report, "demo_data": await _is_demo_data(factory)}
    )


@router.get("/queue", include_in_schema=False)
async def queue_page(request: Request):
    from fairline.db.models import SteamCandidate

    factory = _factory()
    candidates = []
    if factory:
        async with factory() as session:
            candidates = (
                (
                    await session.execute(
                        select(SteamCandidate)
                        .where(SteamCandidate.status == "pending")
                        .order_by(desc(SteamCandidate.created_at))
                        .limit(100)
                    )
                )
                .scalars()
                .all()
            )
    return templates.TemplateResponse(
        request, "queue.html", {"page": "queue", "candidates": candidates, "demo_data": await _is_demo_data(factory)}
    )


@router.get("/record", include_in_schema=False)
async def record_page(request: Request):
    from fairline.db.models import Pick

    factory = _factory()
    picks = []
    if factory:
        async with factory() as session:
            picks = (
                (
                    await session.execute(
                        select(Pick)
                        .where(Pick.result.is_not(None))
                        .where(Pick.user_id == PUBLIC_RECORD_USER_ID)
                        .order_by(desc(Pick.approved_at))
                        .limit(200)
                    )
                )
                .scalars()
                .all()
            )
    settled_clv = [p.clv for p in picks if p.clv is not None]
    totals = {
        "count": len(picks),
        "avg_clv": sum(settled_clv) / len(settled_clv) if settled_clv else 0.0,
        "units": sum(p.profit_units or 0.0 for p in picks),
    }
    return templates.TemplateResponse(
        request, "record.html", {"page": "record", "picks": picks, "totals": totals, "demo_data": await _is_demo_data(factory)}
    )
