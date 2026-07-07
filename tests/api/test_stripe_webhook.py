"""Stripe webhook dedup and safe field access.

A redelivered event (same event id) must be ignored so it cannot flip is_pro a
second time, and a missing customer field must not raise. These call the handler
directly with stripe verification and the DB session factory stubbed out.
"""

from __future__ import annotations

import pytest

import steambot.api.main  # noqa: F401 -- defines get_graph before routes import
from steambot.api import routes


class _FakeRequest:
    async def body(self) -> bytes:
        return b"{}"


class _FakeSession:
    def __init__(self, counter):
        self._counter = counter

    async def __aenter__(self):
        self._counter["opened"] += 1
        return self

    async def __aexit__(self, *args):
        return False

    async def execute(self, *args, **kwargs):
        class _Result:
            def scalar_one_or_none(self_inner):
                return None

        return _Result()

    async def commit(self):
        pass


@pytest.fixture(autouse=True)
def clear_events():
    routes._processed_stripe_events.clear()
    yield
    routes._processed_stripe_events.clear()


@pytest.fixture
def stub_stripe_and_db(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    counter = {"opened": 0}

    def fake_factory():
        return lambda: _FakeSession(counter)

    monkeypatch.setattr(routes, "get_session_factory", fake_factory)
    return counter


def _event(event_id: str, event_type: str, customer: str | None = "cus_123"):
    obj = {} if customer is None else {"customer": customer}
    return {"id": event_id, "type": event_type, "data": {"object": obj}}


async def test_repeated_event_id_is_ignored(stub_stripe_and_db, monkeypatch):
    counter = stub_stripe_and_db
    monkeypatch.setattr(
        routes.stripe.Webhook,
        "construct_event",
        lambda payload, sig, secret: _event("evt_1", "customer.subscription.created"),
    )

    first = await routes.stripe_webhook(_FakeRequest(), stripe_signature="sig")
    assert first == {"received": True}
    assert counter["opened"] == 1

    second = await routes.stripe_webhook(_FakeRequest(), stripe_signature="sig")
    assert second == {"received": True, "duplicate": True}
    # DB session must not open again for the redelivered event.
    assert counter["opened"] == 1


async def test_missing_customer_field_skips_lookup(stub_stripe_and_db, monkeypatch):
    counter = stub_stripe_and_db
    monkeypatch.setattr(
        routes.stripe.Webhook,
        "construct_event",
        lambda payload, sig, secret: _event(
            "evt_2", "customer.subscription.created", customer=None
        ),
    )

    result = await routes.stripe_webhook(_FakeRequest(), stripe_signature="sig")
    assert result == {"received": True}
    # No customer id, so no user lookup should have happened.
    assert counter["opened"] == 0
