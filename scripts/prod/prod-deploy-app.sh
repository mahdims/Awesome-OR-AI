#!/usr/bin/env bash
# Deploy the FastAPI app to prod.
#
# This is a stub during M1a. M1b lands the FastAPI service named `api` in
# /opt/researchmate/docker-compose.yml; at that point this script becomes:
#
#   ssh "$PROD_HOST" '
#     cd /opt/researchmate/app && git pull &&
#     docker compose -f /opt/researchmate/docker-compose.yml build api &&
#     docker compose -f /opt/researchmate/docker-compose.yml run --rm api alembic upgrade head &&
#     docker compose -f /opt/researchmate/docker-compose.yml up -d --no-deps api &&
#     curl -fs https://app.researchmate.app/health/deep
#   '
#
# Build-on-server, no registry — see plan for rationale.
#
# Usage:
#   bash scripts/prod/prod-deploy-app.sh             # real deploy (M1b+)
#   bash scripts/prod/prod-deploy-app.sh --dry-run   # print what would run

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# Probe: does the `api` service exist in prod's compose file yet?
if prod_ssh "docker compose -f /opt/researchmate/docker-compose.yml config --services 2>/dev/null | grep -qx api"; then
  HAS_API=1
else
  HAS_API=0
fi

if [[ $HAS_API -eq 0 ]]; then
  cat <<EOF
[prod-deploy-app] [M1b deferred] No 'api' service in /opt/researchmate/docker-compose.yml
[prod-deploy-app] on $PROD_HOST yet. The placeholder 'web' (nginxdemos/hello) is still in place.
[prod-deploy-app] Re-run this script once M1b adds the FastAPI container.
EOF
  exit 0
fi

# M1b+: real deploy.
if [[ $DRY_RUN -eq 1 ]]; then
  cat <<EOF
[prod-deploy-app] --dry-run: would run on $PROD_HOST:
  cd $PROD_APP_DIR && git pull
  docker compose -f /opt/researchmate/docker-compose.yml build api
  docker compose -f /opt/researchmate/docker-compose.yml run --rm api alembic upgrade head
  docker compose -f /opt/researchmate/docker-compose.yml up -d --no-deps api
  curl -fs https://app.researchmate.app/health/deep
EOF
  exit 0
fi

prod_ssh bash -s -- "$PROD_APP_DIR" <<'REMOTE'
set -euo pipefail
PROD_APP_DIR="$1"

cd "$PROD_APP_DIR" && git pull

cd /opt/researchmate
docker compose build api
docker compose run --rm api alembic upgrade head
docker compose up -d --no-deps api

# Health check from inside the box (Caddy routes externally).
sleep 3
curl -fs https://app.researchmate.app/health/deep
REMOTE

echo "[prod-deploy-app] Done."
