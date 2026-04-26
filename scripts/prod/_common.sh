# Shared bootstrap for scripts/prod/*. Sourced, never executed.
# Loads .env.prod (laptop-only, gitignored) and validates required vars.

# Resolve script dir even when this is sourced from a different cwd.
_PROD_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[1]}" )" &> /dev/null && pwd )"
_PROD_ENV_FILE="$_PROD_SCRIPT_DIR/.env.prod"

if [[ ! -f "$_PROD_ENV_FILE" ]]; then
  echo "[prod] $_PROD_ENV_FILE not found." >&2
  echo "       Copy .env.prod.example to .env.prod and fill in." >&2
  exit 2
fi

# shellcheck disable=SC1090
set -a; source "$_PROD_ENV_FILE"; set +a

: "${PROD_HOST:?[prod] PROD_HOST not set in .env.prod}"
: "${PROD_APP_DIR:?[prod] PROD_APP_DIR not set in .env.prod}"

# LOCAL_DATABASE_URL is only required by prod-data-push.sh; check there.

# Convenience wrapper so script call sites read top-down.
prod_ssh() { ssh -o BatchMode=yes "$PROD_HOST" "$@"; }
