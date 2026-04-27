# Researchmate API image.
#
# Single image is what runs in local dev (with bind-mount + uvicorn --reload)
# and what runs on prod (no bind-mount, no reload). The compose file decides
# which mode by overriding the command + adding a volume in dev only.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Postgres client tools are useful for ad-hoc psql/pg_dump from inside the
# container during ops; small footprint, worth keeping in the image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt

COPY . .

EXPOSE 80

# Default command — overridden in compose for dev (--reload) and prod (workers).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]
