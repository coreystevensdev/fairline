# Matchup agent design

The filter workflow this replaces: a bettor opens a props tool, picks a line,
and slices the player's history by matchup similarity (vs top-10 defenses,
home/away, last N, division games) before deciding. The matchup agent runs
that loop itself: code computes the splits, the LLM selects which ones matter
for this prop and writes the rationale, and every output is CLV-graded like
any other agent's picks.

Depends on props P1 (prop odds and devig) and the player stats ingest. Both
ship before this agent.

## Principle

Same rule as the sim agent: the LLM never produces a probability. Hit rates
are computed; the probability that reaches the blend comes from a shrinkage
model over those hit rates; Claude chooses salient angles and narrates.

## Data

- Player game logs: nflverse weekly player stats (free CSV releases; the same
  distribution channel as the games file the backfill already uses).
- Opponent context per game: defense rank vs position, pace, home/away, rest
  days, division flag. Derivable from the same logs plus game_results.
- Prop lines and prices: props P1 tables.

## The splits engine (pure code)

For a prop (player, market, line), compute a fixed, pre-registered set of
splits, each returning (hits, attempts) against the current line:

- last 5 and last 10 games
- home / away
- vs similar defenses (opponent rank vs position bucketed into thirds)
- division opponents
- with / without a designated teammate active (phase 2 of this doc)

Pre-registered means the set is fixed in code before the season, not chosen
per prop. Letting anything search for the best-looking split is the
multiple-comparisons trap that makes every prop "8 of the last 10" at
something. New splits can be added, but they start unproven (see validation).

## Shrinkage

Raw hit rates are noise at these sample sizes. Each split's rate is shrunk
toward the league base rate for that market with a beta-binomial prior:

    shrunk = (hits + k * base_rate) / (attempts + k)

with k around 10. An 8-of-10 at a 50% base rate becomes 65%, not 80%. The
matchup probability for the prop is a precision-weighted combination of the
shrunk splits, capped so no single split dominates.

## The agent

Node or on-demand endpoint (decided at build time; props are volume, so
on-demand per reviewed prop is the likely v1). Inputs: the prop, its devigged
fair probability, the computed splits. Claude receives the splits as data and
returns which angles are relevant and a rationale citing them; code assembles
the final probability from the shrunk splits it selected, bounded to within
a fixed distance of the market's fair probability.

Output: prop candidates with source="matchup", entering the same review,
settlement (closing prop line), grading (box scores), and leaderboard.

## Validation: per-angle CLV

The extension of the agent leaderboard down one level: every pick records
which splits fed it, and settlement attributes CLV per split. A season of
data answers "does the vs-similar-defense angle actually beat closing prop
lines?" per angle. Angles with non-positive CLV over a real sample get their
weight zeroed. No manual filter user audits their filters; this is the part
only the automated version can do.

## Out of scope for v1

Injury-aware usage projections, live props, same-game parlay correlations,
and any sport beyond NFL. Teammate-active splits wait for reliable
availability data.
