# Simulation agent design

The sim layer today blends caller-supplied probabilities into pick edge (see
"Blending an external simulation" in the README). This document specifies the
agent that replaces the caller as the source of those probabilities: a
deterministic model computed in code, fed by public data, with the LLM kept
out of the arithmetic.

## Principle

The model's numbers come from math over data; Claude never estimates a
probability. LLM probability estimates are poorly calibrated and carry no
information past the training cutoff. Where an LLM helps (reading injury
reports, summarizing rationale), it extracts facts; code converts facts to
numbers. This is the same rule pick_agent already follows for edge and EV.

## Architecture

A `sim_agent` node runs between `odds_agent` and `pick_agent`:

```
odds_agent -> sim_agent -> pick_agent -> hitl_review -> validate_agent
```

It reads the day's games from state, computes per-market probabilities, and
writes them as `sim_lines`, the same state key the API request populates
today. Caller-supplied sims remain supported and override the agent's values
for matching selections, so an operator with a better model keeps the last
word. Everything downstream (blend weight, `sim_probability` persistence,
`sim-report` CLV buckets) is already built and does not change.

## Data source

nflverse (the nflfastR project) publishes complete NFL schedules and results
as versioned CSV releases on GitHub. Free, no API key, community-maintained,
and the standard substrate for public NFL analytics. Phase 1 needs only game
results: date, teams, final scores. Ingest is one HTTP download cached
locally, refreshed weekly during the season.

## Phase 1: ratings and the margin model (h2h, spreads)

Team strength is a single points-scale rating fit from historical results:

- Rating update: Elo-style, margin-aware. Expected margin for a game is
  `rating_home - rating_away + HFA` with home-field advantage around 2.0
  points. After each result, ratings move toward the observed margin with a
  learning rate small enough that one blowout does not swing a season.
- Margin distribution: final margin is modeled as Normal(expected_margin,
  sigma) with sigma near 13.5, the long-run NFL value.
- Win probability: P(margin > 0).
- Spread cover probability: P(margin > line) for the favorite side.

All pure functions over a ratings table. Tests pin known cases: equal teams
at home win about 56% of the time; a 7-point favorite covers -3 well over
half the time.

Known modeling error, accepted for v1: the normal approximation ignores key
numbers. Real NFL margins spike at 3 and 7, so push and near-push
probabilities are misestimated near those lines. A discrete margin
distribution fit from historical margins is the eventual fix and slots in
behind the same function signature.

## Phase 2: totals

Team offensive and defensive scoring rates from the same results data,
opponent-adjusted, combined into an expected game total. Total score modeled
as Normal(expected_total, sigma near 10); over probability is P(total >
line). Reuses Phase 1's ingestion and distribution helpers.

## Phase 3: news adjustments (research agent)

The only inputs where speed can beat the market are injuries and lineups.
A research agent fetches current injury reports, and Claude extracts
structured facts from them: player, team, status. Code then applies bounded,
quantified adjustments to the Phase 1 ratings before simulation. Market
convention anchors the magnitudes: a starting quarterback is worth roughly
5 to 7 points; most other single players under 1.5.

Two hard rules: adjustments are capped in code regardless of what the LLM
extracted, and every adjustment is logged onto the pick's rationale so the
HITL reviewer sees why the number moved.

## Validation

The sim earns blend weight only through `sim-report`: average CLV on picks
where the sim disagreed with the sharp line by 2 or more probability points.
A ratings model built from public results will agree with Pinnacle most of
the time, and the agreed bucket proves nothing. If the disagreed bucket does
not show positive CLV over a meaningful sample, the correct response is to
lower `FAIRLINE_SIM_WEIGHT`, not to tune the model until it agrees with
itself.

## Out of scope

Player-level projection models, live in-game probabilities, and any sport
beyond NFL. Multi-sport waits until the NFL loop has a graded CLV sample.
