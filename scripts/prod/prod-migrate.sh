#!/usr/bin/env bash
# Apply pending Alembic migrations to prod Postgres.
#
# Runs alembic in an ephemeral python:3.13-slim container attached to the
# prod Docker network. No host-side Python install needed, no port exposure.
#
# Prerequisite: repo cloned at $PROD_APP_DIR on the box (M1b onward).
# Until that exists this script exits with a clear message — for M1a today
# the only migration (0001_initial_schema) is already on prod via pg_restore.
#
# Usage:
#   bash scripts/prod/prod-migrate.sh             # alembic upgrade head
#   bash scripts/prod/prod-migrate.sh --dry-run   # alembic current (no writes)

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

ALEMBIC_CMD="upgrade head"
if [[ $DRY_RUN -eq 1 ]]; then
  ALEMBIC_CMD="current"
fi

# Verify the repo exists on the box before launching the migrator container,
# so we get a clean error instead of a Python ImportError.
if ! prod_ssh "test -d $PROD_APP_DIR/src/db/migrations"; then
  cat >&2 <<EOF
[prod-migrate] $PROD_APP_DIR/src/db/migrations not found on $PROD_HOST.
[prod-migrate] First-time setup (run once when M1b is ready):
[prod-migrate]
[prod-migrate]   ssh $PROD_HOST '
[prod-migrate]     git clone https://github.com/mahdims/Awesome-OR-AI.git $PROD_APP_DIR
[prod-migrate]   '
[prod-migrate]
[prod-migrate] For M1a today: the only migration (0001_initial_schema) is
[prod-migrate] already on prod via the 2026-04-26 pg_restore. Nothing to do.
EOF
  exit 1
fi

echo "[prod-migrate] Running 'alembic $ALEMBIC_CMD' on $PROD_HOST..."

prod_ssh bash -s -- "$PROD_APP_DIR" "$ALEMBIC_CMD" <<'REMOTE'
set -euo pipefail
PROD_APP_DIR="$1"
ALEMBIC_CMD="$2"

docker run --rm \
  --network researchmate_internal \
  -v "$PROD_APP_DIR":/app -w /app \
  --env-file /opt/researchmate/.env \
  -e DATABASE_URL='postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}' \
  python:3.13-slim bash -c "
    pip install -q alembic 'psycopg[binary]>=3.2' 'pgvector>=0.3' 'python-dotenv>=1.0' &&
    alembic $ALEMBIC_CMD
  "
REMOTE

echo "[prod-migrate] Done."
