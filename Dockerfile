# Build the virtualenv here, ship only the virtualenv. uv is 37MB and the sync
# layer carries build residue, none of which the running app needs.
FROM python:3.12-slim AS builder

# Pinned rather than :latest so the image is reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.8.9 /uv /usr/local/bin/uv

WORKDIR /app

# Copy mode because there is no cache mount to hardlink against. Bytecode is
# deliberately not precompiled: it costs about 90MB, and the instance these
# demos share has 16GiB for six of them and filled up once already.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0

# src/ has to land before the install. pyproject points hatchling at
# src/fairline, and installing first registers the package against a directory
# that does not exist yet: the install succeeds, the build goes green, and the
# container dies on `No module named 'fairline'` when alembic imports models.
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Installs from uv.lock, so the container holds exactly the tree the lock records
# and the dependency graph scans, rather than whatever pip resolved from
# pyproject on the day of the build.
#
# --locked and not --frozen. Both install from the lock; only --locked fails when
# the lock no longer matches pyproject. Measured, since the names suggest the
# reverse.
#
# --no-editable, where this was `pip install -e .`. The editable install pointed
# at src/, which is why that directory had to be copied; the venv now carries the
# package itself, so the runtime stage below does not need src/ at all.
RUN uv sync --locked --no-editable

FROM python:3.12-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv

# alembic is a runtime dependency here, not a dev one: the CMD migrates before
# it serves, so the migration tree ships.
COPY alembic.ini .
COPY alembic/ alembic/

CMD ["sh", "-c", "alembic upgrade head && uvicorn fairline.api.main:app --host 0.0.0.0 --port 8000"]
