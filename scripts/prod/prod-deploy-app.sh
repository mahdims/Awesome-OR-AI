#!/usr/bin/env bash
# Deploy the FastAPI app to prod.
#
# Builds on the server, no registry (see plan rationale). Assumes the operator
# has already done the one-time cutover from docs/PROD_M1B_CUTOVER.md:
#   - /opt/researchmate/docker-compose.yml has an `api` service block
#   - /opt/researchmate/caddy/Caddyfile points reverse_proxy at api:80
#
# Run alembic migrations BEFORE recreating the service so the schema is at
# head when the new image starts serving requests.
#
# Usage:
#   bash scripts/prod/prod-deploy-app.sh             # real deploy
#   bash scripts/prod/prod-deploy-app.sh --dry-run   # print what would run

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# Probe: does the `api` service exist in prod's compose file?
if ! prod_ssh "docker compose -f /opt/researchmate/docker-compose.yml config --services 2>/dev/null | grep -qx api"; then
  cat >&2 <<EOF
[prod-deploy-app] No 'api' service in /opt/researchmate/docker-compose.yml on $PROD_HOST.
[prod-deploy-app] One-time setup: see docs/PROD_M1B_CUTOVER.md, then re-run.
EOF
  exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
  cat <<EOF
[prod-deploy-app] --dry-run: would run on $PROD_HOST:
  cd $PROD_APP_DIR && git pull
  docker compose -f /opt/researchmate/docker-compose.yml build api
  docker compose -f /opt/researchmate/docker-compose.yml up -d --no-deps api
  curl -fs https://app.researchmate.app/health/deep
EOF
  exit 0
fi

echo "[prod-deploy-app] Pulling repo + building image on $PROD_HOST..."
REMOTE_CMD=$(cat <<EOS
set -euo pipefail
cd $PROD_APP_DIR && git pull --ff-only

cd /opt/researchmate
docker compose build api
docker compose up -d --no-deps api

# Wait for the api container to come up + report fully healthy.
# /health/deep returns 200 even when degraded (db:false), so curl -fs alone
# would let a half-broken deploy through — we have to inspect the body.
healthy=0
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 2
  body=\$(curl -fsS http://127.0.0.1:8080/health/deep 2>/dev/null || true)
  echo "  [\$i/10] /health/deep: \$body"
  if [[ "\$body" == *'"db":true'* && "\$body" == *'"redis":true'* ]]; then
    healthy=1
    break
  fi
done
if [[ \$healthy -ne 1 ]]; then
  echo "[remote] FAIL: api never reported db+redis healthy after 20s. Rolling back not automatic — investigate." >&2
  exit 1
fi

# Public-facing health check via Caddy. Same body check applies.
public_body=\$(curl -fsS https://app.researchmate.app/health/deep 2>/dev/null || true)
echo "  public /health/deep: \$public_body"
if [[ "\$public_body" != *'"db":true'* || "\$public_body" != *'"redis":true'* ]]; then
  echo "[remote] FAIL: public /health/deep not fully healthy." >&2
  exit 1
fi
EOS
)

prod_ssh "$REMOTE_CMD"
echo "[prod-deploy-app] Done."
