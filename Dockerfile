FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY alembic.ini .
COPY alembic/ alembic/

CMD ["sh", "-c", "alembic upgrade head && uvicorn steambot.api.main:app --host 0.0.0.0 --port 8000"]
