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
