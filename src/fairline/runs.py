"""Run registry: Postgres-backed, with an in-memory fallback for demo mode.

Only status and ownership live here; graph state stays in the LangGraph
checkpointer. With a database, runs survive restarts and are visible across
processes. Without one (no DATABASE_URL), a module-level dict preserves the
single-process demo exactly as before.
"""

from __future__ import annotations

from sqlalchemy import select

from fairline.db.models import Run

_memory_runs: dict[str, dict] = {}


async def create_run(session_factory, run_id: str, user_id: str) -> None:
    if session_factory is None:
        _memory_runs[run_id] = {"run_id": run_id, "user_id": user_id, "status": "running", "error": None}
        return
    async with session_factory() as session:
        session.add(Run(id=run_id, user_id=user_id, status="running"))
        await session.commit()


async def fetch_run(session_factory, run_id: str) -> dict | None:
    if session_factory is None:
        return _memory_runs.get(run_id)
    async with session_factory() as session:
        run = (
            (await session.execute(select(Run).where(Run.id == run_id))).scalars().one_or_none()
        )
    if run is None:
        return None
    return {"run_id": run.id, "user_id": run.user_id, "status": run.status, "error": run.error}


async def claim_run(session_factory, run_id: str, from_status: str, to_status: str) -> bool:
    """Atomically move a run between statuses; False means someone else won.

    Collapses the check-then-resume race on approve: two concurrent requests
    both read awaiting_review, but only one flips it.
    """
    if session_factory is None:
        record = _memory_runs.get(run_id)
        if not record or record["status"] != from_status:
            return False
        record["status"] = to_status
        return True
    from sqlalchemy import update as sa_update

    async with session_factory() as session:
        result = await session.execute(
            sa_update(Run)
            .where(Run.id == run_id, Run.status == from_status)
            .values(status=to_status)
        )
        await session.commit()
        return result.rowcount == 1


async def update_run(session_factory, run_id: str, status: str, error: str | None = None) -> None:
    if session_factory is None:
        record = _memory_runs.get(run_id)
        if record:
            record["status"] = status
            record["error"] = error
        return
    async with session_factory() as session:
        run = (
            (await session.execute(select(Run).where(Run.id == run_id))).scalars().one_or_none()
        )
        if run:
            run.status = status
            run.error = error
            await session.commit()
