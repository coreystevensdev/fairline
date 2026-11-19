FROM python:3.12-slim AS base

WORKDIR /app

# src/ has to land before the editable install. pyproject points hatchling at
# src/fairline, and installing first registers the package against a directory
# that does not exist yet: the install succeeds, the build goes green, and the
# container dies on `No module named 'fairline'` when alembic imports models.
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY alembic.ini .
COPY alembic/ alembic/

CMD ["sh", "-c", "alembic upgrade head && uvicorn fairline.api.main:app --host 0.0.0.0 --port 8000"]
