#!/usr/bin/env bash
# Push laptop's Postgres data to production.
#
# Use this after a local pipeline run produces new papers/analyses and you
# want prod to mirror local. Destructive on prod — drops + re-creates the
# database from a fresh pg_dump.
#
# Compatibility:
#   The dump must be readable by prod's Postgres 16. macOS's libpq Homebrew
#   formula tracks major Postgres releases (libpq 18.x writes a v1.16 custom
#   dump format that pg16 can't read), so we pin pg_dump to postgresql@16.
#   Override with $PG_DUMP_BIN if your setup differs.
#
# Safety:
#   - Requires typing RESTORE to confirm.
#   - Aborts if local has fewer paper_analyses rows than prod (catches
#     accidental clobber with an empty laptop DB).
#   - Asserts post-restore row counts on prod ≥ pre-restore prod counts.
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

# psql/pg_dump don't understand the SQLAlchemy "+psycopg" suffix the rest of
# the codebase uses; strip it for the libpq tools.
LOCAL_PG_URL="${LOCAL_DATABASE_URL/postgresql+psycopg:\/\//postgresql://}"

# Find pg_dump / psql.
#
#   1. Honor explicit PG_DUMP_BIN / PSQL_BIN if set (escape hatch — used as-is).
#   2. Otherwise prefer macOS Homebrew's postgresql@16 keg (matches prod's pg16
#      and keeps libpq 18.x — installed for psql/pg_dump-via-PATH-elsewhere —
#      from writing dump headers prod can't read).
#   3. Otherwise fall back to whatever's on $PATH (Linux package managers
#      typically ship a major-matched client, and any compatible build works).

_HOMEBREW_PG16_BIN="/opt/homebrew/opt/postgresql@16/bin"

_resolve_bin() {
  local name="$1" override="$2"
  if [[ -n "$override" ]]; then
    if [[ ! -x "$override" ]]; then
      echo "[prod-data-push] override path not executable: $override" >&2
      return 1
    fi
    echo "$override"
    return 0
  fi
  if [[ -x "$_HOMEBREW_PG16_BIN/$name" ]]; then
    echo "$_HOMEBREW_PG16_BIN/$name"
    return 0
  fi
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return 0
  fi
  return 1
}

PG_DUMP_BIN="$(_resolve_bin pg_dump "${PG_DUMP_BIN:-}")" || {
  echo "[prod-data-push] pg_dump not found." >&2
  echo "[prod-data-push] Install pg_dump (macOS: brew install postgresql@16; debian: apt install postgresql-client-16) or set PG_DUMP_BIN." >&2
  exit 1
}
PSQL_BIN="$(_resolve_bin psql "${PSQL_BIN:-}")" || {
  echo "[prod-data-push] psql not found." >&2
  exit 1
}

# Major-version sanity check — pg16 server can't read pg18-format dumps.
_PG_DUMP_MAJOR=$("$PG_DUMP_BIN" --version 2>/dev/null | grep -oE '[0-9]+' | head -1)
if [[ "$_PG_DUMP_MAJOR" != "16" ]]; then
  echo "[prod-data-push] WARN: pg_dump is major version ${_PG_DUMP_MAJOR:-unknown}; prod runs pg16." >&2
  echo "[prod-data-push] WARN: dump may be unreadable by prod. Continuing — set PG_DUMP_BIN to override." >&2
fi
echo "[prod-data-push] using pg_dump=$PG_DUMP_BIN (major=$_PG_DUMP_MAJOR), psql=$PSQL_BIN"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# Compare row counts so we don't clobber prod with a smaller local DB.
echo "[prod-data-push] Checking row counts..."
LOCAL_COUNT=$("$PSQL_BIN" "$LOCAL_PG_URL" -tAc 'SELECT count(*) FROM paper_analyses')
PROD_COUNT=$(prod_ssh "docker exec researchmate_postgres psql -U researchmate -d researchmate -tAc 'SELECT count(*) FROM paper_analyses'")
echo "[prod-data-push]   local paper_analyses: $LOCAL_COUNT"
echo "[prod-data-push]   prod  paper_analyses: $PROD_COUNT"

if (( LOCAL_COUNT < PROD_COUNT )); then
  echo "[prod-data-push] REFUSING: local has fewer rows than prod." >&2
  echo "[prod-data-push] If this is intentional, restore from R2 instead." >&2
  exit 1
fi

DUMP_FILE="/tmp/researchmate-$(date -u +%Y%m%dT%H%M%SZ).dump"

cat <<EOF
[prod-data-push] Plan:
  pg_dump  ($PG_DUMP_BIN, format=custom)  $LOCAL_PG_URL  ->  $DUMP_FILE
  scp      $DUMP_FILE  ->  $PROD_HOST:$DUMP_FILE
  remote: stop api, dropdb+createdb, pg_restore, start api, verify counts
  remote: rm $DUMP_FILE   (and locally too on success)
EOF

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[prod-data-push] --dry-run: stopping here."
  exit 0
fi

read -r -p "[prod-data-push] Type RESTORE to proceed: " CONFIRM
if [[ "$CONFIRM" != "RESTORE" ]]; then
  echo "[prod-data-push] Aborted."
  exit 1
fi

# Dump + ship.
echo "[prod-data-push] Dumping local with pg16 client..."
"$PG_DUMP_BIN" "$LOCAL_PG_URL" -F c -f "$DUMP_FILE"
DUMP_SIZE=$(du -h "$DUMP_FILE" | cut -f1)
echo "[prod-data-push] Dump size: $DUMP_SIZE"

echo "[prod-data-push] Uploading to $PROD_HOST..."
scp "$DUMP_FILE" "$PROD_HOST:$DUMP_FILE"

# Build the remote command as a single string and send it via `ssh host '...'`
# (single-arg form). Avoids `bash -s <<heredoc` because docker exec -i can race
# for SSH stdin and silently truncate the script.
echo "[prod-data-push] Restoring on prod..."
REMOTE_CMD=$(cat <<EOS
set -euo pipefail
DUMP_FILE='$DUMP_FILE'

cd /opt/researchmate

# Stop the api so it doesn't choke on the DB being recreated underneath it.
# After M1b cutover the placeholder \`web\` is gone — it's \`api\` now.
docker compose stop api

docker exec researchmate_postgres dropdb   -U researchmate researchmate --force
docker exec researchmate_postgres createdb -U researchmate researchmate

# Copy dump file into the container, then pg_restore from it. We do this
# instead of piping with -i because docker exec -i and SSH stdin can race.
docker cp "\$DUMP_FILE" researchmate_postgres:/tmp/restore.dump
docker exec researchmate_postgres pg_restore \\
  -U researchmate -d researchmate --no-owner --role=researchmate \\
  /tmp/restore.dump
docker exec researchmate_postgres rm /tmp/restore.dump

rm "\$DUMP_FILE"

# Bring api back up. Migrations: any new schema landed since the dump was
# taken would be re-applied by the next prod-deploy-app.sh run, but typical
# usage is that this script is run AFTER deploys, so the schema already
# matches.
docker compose start api

# Assert: prod row count >= what it was before
NEW_COUNT=\$(docker exec researchmate_postgres psql -U researchmate -d researchmate -tAc 'SELECT count(*) FROM paper_analyses')
echo "[remote] paper_analyses after restore: \$NEW_COUNT"
if [[ "\$NEW_COUNT" -lt $PROD_COUNT ]]; then
  echo "[remote] FAIL: post-restore count \$NEW_COUNT < pre-restore $PROD_COUNT" >&2
  exit 1
fi
echo "[remote] OK: rows preserved or grown."
EOS
)

prod_ssh "$REMOTE_CMD"

# Local cleanup.
rm "$DUMP_FILE"
echo "[prod-data-push] Done."