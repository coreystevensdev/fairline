"""CLI wiring test: --sport baseball_mlb routes to the MLB matchup path,
not the NFL one."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_matchup_dispatches_to_mlb_path_for_mlb_sport(monkeypatch):
    import fairline.__main__ as cli_module

    called_with = {}

    async def fake_create_mlb_matchup_candidates(session_factory, snapshot, min_edge):
        called_with["snapshot_sport"] = snapshot.sport
        return 2

    async def fake_create_matchup_candidates(session_factory, snapshot, min_edge):
        called_with["nfl_path_called"] = True
        return 0

    monkeypatch.setattr(
        "fairline.mlb_matchup.create_mlb_matchup_candidates", fake_create_mlb_matchup_candidates
    )
    monkeypatch.setattr(
        "fairline.matchup.create_matchup_candidates", fake_create_matchup_candidates
    )

    async def fake_fetch_odds(client, sport):
        from datetime import datetime, timedelta, timezone

        from fairline.state import GameSnapshot

        return [
            GameSnapshot(
                game_id="g1", sport=sport, home_team="New York Yankees",
                away_team="Boston Red Sox",
                commence_time=datetime.now(timezone.utc) + timedelta(hours=2),
                bookmakers=[],
            )
        ]

    async def fake_fetch_event_props(client, sport, game_id, markets):
        from fairline.state import GameSnapshot
        from datetime import datetime, timezone

        return GameSnapshot(
            game_id=game_id, sport=sport, home_team="New York Yankees",
            away_team="Boston Red Sox", commence_time=datetime.now(timezone.utc), bookmakers=[],
        )

    monkeypatch.setattr(cli_module, "_fetch_odds_for", lambda client, sport: fake_fetch_odds(client, sport))
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr("fairline.clients.odds_api.fetch_odds", fake_fetch_odds)
    monkeypatch.setattr("fairline.clients.odds_api.fetch_event_props", fake_fetch_event_props)

    await cli_module._matchup("baseball_mlb", "batter_hits", 0.03, 5)

    assert called_with.get("snapshot_sport") == "baseball_mlb"
    assert "nfl_path_called" not in called_with
