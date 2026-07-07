"""Tests for session factory resolution at app boot."""

from __future__ import annotations

import pytest

import fairline.db.session as db_session
from fairline.api.main import resolve_session_factory


@pytest.fixture
def no_cached_engine(monkeypatch):
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_session_factory", None)


async def test_production_without_database_url_refuses_boot(monkeypatch, no_cached_engine):
    monkeypatch.setenv("FAIRLINE_ENV", "production")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        resolve_session_factory()


async def test_dev_without_database_url_returns_none(monkeypatch, no_cached_engine):
    monkeypatch.delenv("FAIRLINE_ENV", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert resolve_session_factory() is None


async def test_production_with_database_url_boots(monkeypatch, no_cached_engine):
    monkeypatch.setenv("FAIRLINE_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

    assert resolve_session_factory() is not None
