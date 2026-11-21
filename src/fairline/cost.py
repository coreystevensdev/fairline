"""Spend guard for pick_agent, the only path in fairline that pays per request."""

from __future__ import annotations

import os

from rolling_cost_cap import CostCap

# claude-sonnet-4-6 list price
ANTHROPIC_INPUT_PER_MTOK: float = 3.00
ANTHROPIC_OUTPUT_PER_MTOK: float = 15.00

MAX_OUTPUT_TOKENS = 4096


def build_cap(monthly_budget: float | None = None) -> CostCap:
    if monthly_budget is None:
        monthly_budget = float(os.environ.get("FAIRLINE_MONTHLY_BUDGET_USD", "10.00"))
    return CostCap(
        # 8 and not 3 because the pre-call estimate is a worst case and the
        # ledger holds actuals, and a terse completion makes that a 4.5x gap.
        # See the pre-call estimate entry in the README's Known limitations.
        multiplier=8.0,
        window=50,
        min_samples=5,
        absolute_ceiling=float(os.environ.get("FAIRLINE_CALL_CEILING_USD", "0.40")),
        monthly_budget=monthly_budget,
    )


# One cap for the process. Per-call would cap nothing.
generation_cap = build_cap()


def call_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1e6 * ANTHROPIC_INPUT_PER_MTOK
        + output_tokens / 1e6 * ANTHROPIC_OUTPUT_PER_MTOK
    )


def worst_case(prompt: str) -> float:
    """The most this call could be billed, for asking before paying."""
    return call_cost(len(prompt) // 4, MAX_OUTPUT_TOKENS)
