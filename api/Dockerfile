# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime deps from pyproject.toml (no dev extras in the image)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Pandoc — the DOCX canonical-stream reader (ADR 0017), invoked at arm's
# length as a subprocess by app/pipeline/parsers.py. Pinned to an exact
# version: Pandoc output is deterministic per version only, and
# documents.parser_version records the version for re-ingest-on-drift
# decisions. Installed here for both the api service and the ingest-worker
# (both build from this Dockerfile).
ARG PANDOC_VERSION=3.6.4
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && arch="$(dpkg --print-architecture)" \
 && curl -fsSL "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-linux-${arch}.tar.gz" \
    | tar -xz -C /usr/local --strip-components=1 "pandoc-${PANDOC_VERSION}/bin/pandoc" \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Copy application code, migration files, and the entrypoint script
COPY app/ ./app/
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/
COPY entrypoint.sh /usr/local/bin/lq-ai-entrypoint
RUN chmod +x /usr/local/bin/lq-ai-entrypoint

EXPOSE 8000

# The entrypoint runs `alembic upgrade head` before exec'ing uvicorn so a
# fresh deployment lands a fully-migrated schema before the HTTP server
# accepts traffic. Operators handling migrations out-of-band can set
# LQ_AI_SKIP_MIGRATIONS=1 to bypass the step.
ENTRYPOINT ["/usr/local/bin/lq-ai-entrypoint"]
