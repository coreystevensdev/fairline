"""SteamBot FastAPI application.

Serves the picks API + minimal Jinja2 UI for the HITL approval workflow.
The httpx.AsyncClient and LangGraph checkpointer are shared across requests
via the lifespan context so connections are pooled.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from steambot.graph import build_graph

logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None
_graph = None
_templates: Jinja2Templates | None = None


def resolve_session_factory():
    """Session factory from DATABASE_URL, or None when running without a DB.

    STEAMBOT_ENV=production turns the missing-DB fallback into a boot failure:
    a prod deployment that silently skips pick persistence has no CLV history,
    which defeats the point of the product.
    """
    try:
        from steambot.db.session import get_session_factory
        return get_session_factory()
    except RuntimeError:
        if os.environ.get("STEAMBOT_ENV", "").lower() == "production":
            raise RuntimeError(
                "STEAMBOT_ENV=production requires DATABASE_URL; refusing to boot without pick persistence"
            ) from None
        logger.warning("DATABASE_URL not set; picks will not be persisted")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client, _graph
    _http_client = httpx.AsyncClient()
    _graph = build_graph(_http_client, session_factory=resolve_session_factory())
    yield
    await _http_client.aclose()
    _http_client = None
    _graph = None


app = FastAPI(title="SteamBot", version="0.1.0", lifespan=lifespan)


def get_http_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("HTTP client not initialized -- is the app running?")
    return _http_client


def get_graph():
    if _graph is None:
        raise RuntimeError("Graph not initialized -- is the app running?")
    return _graph


from steambot.api import routes  # noqa: E402 -- import after app is defined

app.include_router(routes.router)
