"""Server-rendered pages: leaderboard, review queue, record."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import fairline.db.session as db_session


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("FAIRLINE_ENV", raising=False)
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_session_factory", None)
    from fairline.api.main import app

    with TestClient(app) as c:
        yield c


def test_leaderboard_is_the_front_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Agent record" in resp.text
    assert "No settled picks yet" in resp.text


def test_queue_page_renders_empty_state(client):
    resp = client.get("/queue")
    assert resp.status_code == 200
    assert "Review queue" in resp.text
    assert "Nothing pending" in resp.text


def test_record_page_renders_empty_state(client):
    resp = client.get("/record")
    assert resp.status_code == 200
    assert "The record" in resp.text


def test_tokens_stylesheet_is_served(client):
    resp = client.get("/static/tokens.css")
    assert resp.status_code == 200
    assert "--accent" in resp.text


@pytest.fixture
def seeded_db(tmp_path):
    """A file-backed sqlite carrying the rows `fairline seed-demo` writes."""
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from fairline.db.models import Base
    from fairline.seed import seed_demo

    url = f"sqlite+aiosqlite:///{tmp_path}/demo.db"

    async def build():
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await seed_demo(async_sessionmaker(engine, expire_on_commit=False))
        await engine.dispose()

    asyncio.run(build())
    return url


@pytest.fixture
def serving_demo_rows(client, monkeypatch, seeded_db):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import fairline.api.pages as pages

    factory = async_sessionmaker(create_async_engine(seeded_db), expire_on_commit=False)
    monkeypatch.setattr(pages, "get_session_factory", lambda: factory)
    return client


@pytest.mark.parametrize("path", ["/", "/queue", "/record"])
def test_seeded_rows_are_labelled_without_anyone_setting_the_flag(
    serving_demo_rows, monkeypatch, path
):
    """The failure this guards against is a graded record that is not real
    being served with no label, which is what an unset flag defaults to."""
    monkeypatch.delenv("FAIRLINE_DEMO_DATA", raising=False)

    resp = serving_demo_rows.get(path)

    assert resp.status_code == 200
    assert "demo-banner" in resp.text, f"{path} served seeded rows with no sample-data label"


def test_the_flag_keeps_the_label_up_after_the_demo_rows_are_gone(client, monkeypatch):
    """An operator who seeded once and now takes real picks keeps saying so."""
    monkeypatch.setenv("FAIRLINE_DEMO_DATA", "true")

    assert "demo-banner" in client.get("/record").text


def test_a_clean_deployment_makes_no_sample_data_claim(client, monkeypatch):
    monkeypatch.delenv("FAIRLINE_DEMO_DATA", raising=False)

    assert "demo-banner" not in client.get("/record").text


def test_record_page_shows_only_the_public_operator(serving_demo_rows, seeded_db, monkeypatch):
    """The schema and the authed API are multi-tenant; these pages are not
    authenticated. Before scoping, /record selected every settled Pick and the
    leaderboard aggregated every user's, so with Stripe subscriptions in the
    repo the second paying user's selections became readable by any visitor.
    Only the demo user had rows, which is what kept it latent."""
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from sqlalchemy import select

    from fairline.db.models import Pick
    from fairline.seed import DEMO_USER_ID

    factory = async_sessionmaker(create_async_engine(seeded_db), expire_on_commit=False)

    async def add_someone_elses_settled_pick():
        async with factory() as session:
            demo = (await session.execute(select(Pick).where(Pick.user_id == DEMO_USER_ID))).scalars().first()
            assert demo is not None, "fixture should have seeded the demo operator's picks"
            clone = Pick(
                **{
                    c.name: getattr(demo, c.name)
                    for c in Pick.__table__.columns
                    if c.name not in {"id", "user_id", "selection"}
                },
                id="other-user-pick",
                user_id="someone-else",
                selection="OTHER USER SELECTION",
            )
            session.add(clone)
            await session.commit()

    asyncio.run(add_someone_elses_settled_pick())

    body = serving_demo_rows.get("/record").text
    assert "OTHER USER SELECTION" not in body, "another user's pick is readable on a public page"
