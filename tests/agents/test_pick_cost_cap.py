"""The one place fairline spends money.

rolling-cost-cap has been a declared dependency since the repo was created and
was imported nowhere, while pick_agent constructed an Anthropic client and
called it with max_tokens=4096. The demo is safe today only because it serves
seeded rows and never reaches this path, which is not the same as a guard.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fairline import cost
from fairline.agents import pick
from fairline.state import FairLine, GameSnapshot

GAME = GameSnapshot(
    game_id="game-1",
    sport="americanfootball_nfl",
    home_team="Kansas City Chiefs",
    away_team="Las Vegas Raiders",
    commence_time=datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc),
    bookmakers=[],
)
FAIR = FairLine(
    game_id="game-1",
    market="h2h",
    outcomes=["Kansas City Chiefs", "Las Vegas Raiders"],
    fair_probs=[0.58, 0.42],
    source_book="pinnacle",
)


def _state():
    return {"games": [GAME], "fair_lines": [FAIR]}


@pytest.fixture(autouse=True)
def fresh_cap(monkeypatch):
    monkeypatch.setattr(cost, "generation_cap", cost.build_cap(monthly_budget=1.00))


class _Usage:
    input_tokens = 3_000
    output_tokens = 900


class _Resp:
    class _Block:
        type = "tool_use"
        name = "submit_picks"
        input = {"picks": []}

    content = [_Block()]
    usage = _Usage()


@pytest.mark.asyncio
async def test_an_exhausted_budget_refuses_before_calling_the_model(monkeypatch):
    called = False

    def must_not_run(*a, **kw):
        nonlocal called
        called = True
        raise AssertionError("the model was called after the budget was spent")

    cost.generation_cap.record(1.00)
    monkeypatch.setattr(pick, "_call_model", must_not_run)

    out = await pick.pick_agent(_state())

    assert called is False
    assert out["candidates"] == []
    assert out["error"] and "budget" in out["error"].lower()


@pytest.mark.asyncio
async def test_the_recorded_cost_is_what_the_call_used(monkeypatch):
    monkeypatch.setattr(pick, "_call_model", lambda prompt: _Resp())
    before = cost.generation_cap.monthly_spend

    await pick.pick_agent(_state())

    spent = cost.generation_cap.monthly_spend - before
    expected = (3_000 / 1e6) * cost.ANTHROPIC_INPUT_PER_MTOK + (900 / 1e6) * cost.ANTHROPIC_OUTPUT_PER_MTOK
    assert spent == pytest.approx(expected, abs=1e-9)


@pytest.mark.asyncio
async def test_a_refusal_is_a_graph_error_not_an_exception(monkeypatch):
    """The graph short-circuits to END on state['error']. Raising instead would
    take down a run that had already paid for four research nodes."""
    def must_not_run(*a, **kw):
        raise AssertionError("called despite an exhausted budget")

    cost.generation_cap.record(1.00)
    monkeypatch.setattr(pick, "_call_model", must_not_run)

    out = await pick.pick_agent(_state())

    assert isinstance(out, dict)
    assert "error" in out


@pytest.mark.asyncio
async def test_the_cap_is_shared_across_runs(monkeypatch):
    """A per-call cap caps nothing: every run would start with a full budget."""
    monkeypatch.setattr(pick, "_call_model", lambda prompt: _Resp())

    await pick.pick_agent(_state())
    after_one = cost.generation_cap.monthly_spend
    await pick.pick_agent(_state())

    assert cost.generation_cap.monthly_spend > after_one


@pytest.mark.asyncio
async def test_terse_completions_do_not_poison_the_rolling_median(monkeypatch):
    """The trap the multiplier exists for.

    The pre-call estimate is the worst case, the ledger holds the actual, and a
    short answer costs about a quarter of the worst case. At the default 3x
    multiplier the sixth call reads its own estimate as an anomaly and refuses
    forever. Six calls here because min_samples is 5.
    """
    class _Terse(_Resp):
        class _U:
            input_tokens = 3_983
            output_tokens = 300
        usage = _U()

    monkeypatch.setattr(pick, "_call_model", lambda prompt: _Terse())

    for _ in range(6):
        out = await pick.pick_agent(_state())
        assert out["error"] is None, f"refused after {cost.generation_cap.monthly_spend:.4f} spent"

    assert cost.generation_cap.median is not None


@pytest.mark.asyncio
async def test_the_budget_is_not_crossed_by_the_call_that_finds_it(monkeypatch):
    """Spend stops below the budget, not above it.

    taxcite probes with check(0.0), so the call that crosses its budget is
    still paid for and only the next one is refused. Estimating first means the
    last call fairline makes is one it could afford.
    """
    monkeypatch.setattr(pick, "_call_model", lambda prompt: _Resp())
    monkeypatch.setattr(cost, "generation_cap", cost.build_cap(monthly_budget=0.10))

    errors = []
    for _ in range(12):
        errors.append((await pick.pick_agent(_state()))["error"])

    assert any(e for e in errors), "the budget was never enforced"
    assert cost.generation_cap.monthly_spend <= 0.10
