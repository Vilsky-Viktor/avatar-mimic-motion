FROM python:3.12-slim

RUN pip install --no-cache-dir poetry

WORKDIR /app

# Install deps via poetry
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root

# Copy source
COPY ingest_job /app/ingest_job

# Run entrypoint
CMD ["poetry", "run", "python", "-m", "ingest_job.main"]