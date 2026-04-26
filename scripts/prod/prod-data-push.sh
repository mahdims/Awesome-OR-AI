#!/usr/bin/env bash
# Push laptop's Postgres data to production.
#
# Use this after a local pipeline run produces new papers/analyses and you
# want prod to mirror local. Destructive on prod — drops + re-creates the
# database from a fresh pg_dump.
#
# Safety:
#   - Requires typing RESTORE to confirm.
#   - Aborts if local has fewer paper_analyses rows than prod (catches
#     accidental clobber with an empty laptop DB).
#   - Last night's prod backup still lives on R2 (latest.dump) for rollback.
#
# Usage:
#   bash scripts/prod/prod-data-push.sh             # real run, interactive confirm
#   bash scripts/prod/prod-data-push.sh --dry-run   # show what would happen, no transfer

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

: "${LOCAL_DATABASE_URL:?[prod-data-push] LOCAL_DATABASE_URL not set in .env.prod}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# 1. Sanity: check pg_dump exists on laptop.
command -v pg_dump >/dev/null || {
  echo "[prod-data-push] pg_dump not found on PATH." >&2
  echo "[prod-data-push] On macOS:  brew install libpq && brew link --force libpq" >&2
  exit 1
}

# 2. Compare row counts so we don't clobber prod with a smaller local DB.
echo "[prod-data-push] Checking row counts..."
LOCAL_COUNT=$(psql "$LOCAL_DATABASE_URL" -tAc 'SELECT count(*) FROM paper_analyses')
PROD_COUNT=$(prod_ssh "docker exec researchmate_postgres psql -U researchmate -d researchmate -tAc 'SELECT count(*) FROM paper_analyses'")
echo "[prod-data-push]   local paper_analyses: $LOCAL_COUNT"
echo "[prod-data-push]   prod  paper_analyses: $PROD_COUNT"

if (( LOCAL_COUNT < PROD_COUNT )); then
  echo "[prod-data-push] REFUSING: local has fewer rows than prod." >&2
  echo "[prod-data-push] If this is intentional, restore from R2 instead." >&2
  exit 1
fi

# 3. Plan summary.
DUMP_FILE="/tmp/researchmate-$(date -u +%Y%m%dT%H%M%SZ).dump"
cat <<EOF
[prod-data-push] Plan:
  pg_dump  $LOCAL_DATABASE_URL  -F c -f $DUMP_FILE
  scp      $DUMP_FILE  $PROD_HOST:$DUMP_FILE
  ssh      $PROD_HOST  '
    docker compose -f /opt/researchmate/docker-compose.yml stop web
    docker exec -i researchmate_postgres dropdb   -U researchmate researchmate --force
    docker exec -i researchmate_postgres createdb -U researchmate researchmate
    cat $DUMP_FILE | docker exec -i researchmate_postgres pg_restore -U researchmate -d researchmate --no-owner --role=researchmate
    rm $DUMP_FILE
    docker compose -f /opt/researchmate/docker-compose.yml start web
  '
EOF

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[prod-data-push] --dry-run: stopping here."
  exit 0
fi

# 4. Confirm.
read -r -p "[prod-data-push] Type RESTORE to proceed: " CONFIRM
if [[ "$CONFIRM" != "RESTORE" ]]; then
  echo "[prod-data-push] Aborted."
  exit 1
fi

# 5. Dump + ship.
echo "[prod-data-push] Dumping local..."
pg_dump "$LOCAL_DATABASE_URL" -F c -f "$DUMP_FILE"
DUMP_SIZE=$(du -h "$DUMP_FILE" | cut -f1)
echo "[prod-data-push] Dump size: $DUMP_SIZE"

echo "[prod-data-push] Uploading to $PROD_HOST..."
scp "$DUMP_FILE" "$PROD_HOST:$DUMP_FILE"

# 6. Restore on box.
echo "[prod-data-push] Restoring on prod..."
prod_ssh bash -s -- "$DUMP_FILE" <<'REMOTE'
set -euo pipefail
DUMP_FILE="$1"

cd /opt/researchmate
docker compose stop web
docker exec -i researchmate_postgres dropdb   -U researchmate researchmate --force
docker exec -i researchmate_postgres createdb -U researchmate researchmate
cat "$DUMP_FILE" | docker exec -i researchmate_postgres pg_restore \
  -U researchmate -d researchmate --no-owner --role=researchmate
rm "$DUMP_FILE"
docker compose start web

echo "[prod-data-push] Restore complete. Row counts:"
docker exec researchmate_postgres psql -U researchmate -d researchmate -c \
  "SELECT (SELECT count(*) FROM papers) AS papers,
          (SELECT count(*) FROM paper_analyses) AS analyses;"
REMOTE

# 7. Local cleanup.
rm "$DUMP_FILE"
echo "[prod-data-push] Done."
