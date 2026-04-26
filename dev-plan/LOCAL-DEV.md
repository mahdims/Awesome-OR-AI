# researchmate — Local Development Setup

**Companion to `INFRA.md`. Read that first.**
**Audience:** anyone (including future-you) standing up a local dev environment.

---

## 1. Goal

A local dev environment that **matches prod close enough that "works on laptop" is a strong signal for "works in prod."** Differences should be deliberate, documented, and limited to:

- No TLS termination locally (Caddy is prod-only; you talk to FastAPI directly)
- No invite-list enforcement locally (trivial bypass for dev convenience)
- Default-on dev passwords (which is fine because nothing is exposed beyond `127.0.0.1`)
- Live-reload mounts so code changes don't require image rebuilds

Everything else — Postgres version, pgvector extension, Redis with auth, env-var names, container hostnames inside the network — must match prod.

---

## 2. Prerequisites

- macOS, Linux, or Windows with WSL2
- Docker Desktop (or Docker Engine + Compose plugin on Linux)
- `git`
- Python 3.12 (or whatever the prod app pins to — match the version in `pyproject.toml` / `Dockerfile`)
- `psql` client on host (optional but useful — `brew install libpq && brew link --force libpq` on Mac)

---

## 3. One-time setup

### 3.1 Clone

```bash
git clone <repo-url> researchmate
cd researchmate
```

### 3.2 `.env` for local dev

Copy `.env.example` to `.env`. **Never commit `.env` itself.** The example file is the contract — it lists every variable the app needs, with safe defaults for local-only use.

```bash
cp .env.example .env
```

A reasonable local `.env`:

```bash
# ── Domain (local) ───────────────────────────────────
# Used in places like email links / OAuth callback origins.
# Local: no TLS, hits FastAPI on port 8000 directly.
APP_BASE_URL=http://localhost:8000

# ── Postgres ─────────────────────────────────────────
POSTGRES_USER=researchmate
POSTGRES_PASSWORD=researchmate_local
POSTGRES_DB=researchmate
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# ── Redis ────────────────────────────────────────────
REDIS_PASSWORD=researchmate_local
REDIS_HOST=redis
REDIS_PORT=6379

# ── App secrets ──────────────────────────────────────
# Replace with `openssl rand -base64 64` if paranoid; for local dev a constant is fine.
JWT_SECRET=local-dev-not-secret-replace-in-prod

# ── Logging / env ────────────────────────────────────
ENV=development
LOG_LEVEL=debug

# ── Auth bypass for local ────────────────────────────
# When true, /auth/google/callback accepts a dev-only header `X-Dev-User: <email>`
# instead of doing a real Google OAuth round-trip. NEVER set true in prod.
DEV_AUTH_BYPASS=true

# ── Optional: real services if you're testing them ───
# GOOGLE_CLIENT_ID=
# GOOGLE_CLIENT_SECRET=
# GEMINI_API_KEY=
# SENTRY_DSN=
# NOTEBOOKLM_SESSION_PATH=./secrets/notebooklm-session.json
```

**Why these specific choices match prod:**
- Same env-var names as `INFRA.md §8`
- Same `POSTGRES_USER` and `POSTGRES_DB` so dump/restore works without rewriting ownership
- `POSTGRES_HOST=postgres` matches the docker-compose service name (works inside containers; on host you'd use `localhost`)
- `JWT_SECRET` is intentionally fixed so existing dev-DB sessions persist across restarts
- `DEV_AUTH_BYPASS` is a code-level flag — described in §9

### 3.3 Local secrets directory

```bash
mkdir -p secrets
```

Some files are too big or stateful for `.env`:

- NotebookLM session cookies (when integrated)
- Local R2 credentials (if you want to test backup script locally)
- TLS certs (not needed locally, but if you ever do)

`secrets/` is in `.gitignore`. Treat it like `.env`.

### 3.4 (Optional) `gcloud`-style credentials cache

If using Gemini locally, set `GEMINI_API_KEY` in `.env` directly. **Use a separate dev API key** with a low monthly cap (`$5`) — don't share the prod key. Set the cap in Google Cloud Console.

---

## 4. The local `docker-compose.yml`

The repo ships TWO compose files:

| File | Purpose |
|---|---|
| `docker-compose.yml` | Production. The one on the box at `/opt/researchmate/`. Includes Caddy, no source code mounts. |
| `docker-compose.dev.yml` | Development. Postgres + Redis + FastAPI + worker. NO Caddy. Mounts source code for live reload. Exposes Postgres/Redis on `127.0.0.1` for direct tooling access. |

By convention, the dev file is the override. Run via:

```bash
docker compose -f docker-compose.dev.yml up -d
```

### 4.1 Reference: `docker-compose.dev.yml`

```yaml
name: researchmate-dev

services:
  postgres:
    image: pgvector/pgvector:pg16            # exact same image as prod
    container_name: researchmate_dev_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - pg_data:/var/lib/postgresql/data
    ports:
      - "127.0.0.1:5432:5432"                # exposed to host ONLY (so psql, GUI tools, alembic from host work)
    networks:
      - internal
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    container_name: researchmate_dev_redis
    restart: unless-stopped
    command: ["redis-server", "--appendonly", "yes", "--requirepass", "${REDIS_PASSWORD}"]
    volumes:
      - redis_data:/data
    ports:
      - "127.0.0.1:6379:6379"                # exposed to host ONLY
    networks:
      - internal
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 10

  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: dev
    container_name: researchmate_dev_api
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./src:/app/src                       # live-reload: edit code, server picks it up
    ports:
      - "127.0.0.1:8000:8000"                # FastAPI direct on host
    networks:
      - internal
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  worker:
    build:
      context: .
      dockerfile: Dockerfile
      target: dev
    container_name: researchmate_dev_worker
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./src:/app/src
    networks:
      - internal
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: ["arq", "src.jobs.worker.WorkerSettings"]

networks:
  internal:
    driver: bridge

volumes:
  pg_data:
  redis_data:
```

### 4.2 How this differs from your current local compose

For reference, your current local file (the one you shared earlier in this project) looks like:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: ri_postgres
    ...
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-ri}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-ri_dev}
      POSTGRES_DB: ${POSTGRES_DB:-research_intelligence}
    ports:
      - "127.0.0.1:5432:5432"
  redis:
    image: redis:7-alpine
    container_name: ri_redis
    ...
    ports:
      - "127.0.0.1:6379:6379"
```

**Where it agrees with prod:**
- ✅ Same Postgres image (`pgvector/pgvector:pg16`)
- ✅ Same Redis image (`redis:7-alpine`)
- ✅ Volumes for data persistence
- ✅ Healthchecks
- ✅ Localhost-only port exposure (`127.0.0.1:`)
- ✅ Redis AOF persistence (`--appendonly yes`)

**Where it diverges and should be reconciled:**

| Item | Your local | Prod | Why prod wins |
|---|---|---|---|
| Container names | `ri_postgres`, `ri_redis` | `researchmate_postgres`, `researchmate_redis` | Naming consistency aids muscle memory and tooling |
| DB user | `ri` | `researchmate` | Prevents needing `--no-owner --role=researchmate` on every restore |
| DB name | `research_intelligence` | `researchmate` | Same string everywhere |
| DB password | `ri_dev` (default in compose) | required from `.env`, no default | Force explicit choice; never accidentally ship a default |
| Redis password | none | required, password-protected | Match prod auth surface |
| Project name | none (default = directory name) | `researchmate-dev` (explicit) | Predictable container/network/volume prefixes |
| Compose env defaults | uses `${VAR:-default}` syntax | uses `${VAR}` (must come from `.env`) | Forces explicit env declaration |

**Migration path (recommended):**

1. Stop your current local stack: `docker compose down` (NOT `down -v` — keeps the data)
2. Rename containers in compose to `researchmate_dev_*`
3. Update env defaults to require `.env`
4. Add `--requirepass` to Redis command
5. Re-import data with the new user/db name (next section)

Or, if you don't want to re-import, keep your current setup and just **document the divergence** — the schema dump/restore step in §5 has an alternate flow for when local user differs from prod.

---

## 5. Importing data into local dev

Three scenarios:

### 5.1 You already have data locally (most common, current state)

Already done. Skip ahead.

### 5.2 You're a new contributor and need a working local DB

Two routes:

**Route A — pull from prod** (quickest if you have SSH access):

```bash
ssh mahdi@5.78.205.155 \
  "docker exec researchmate_postgres pg_dump -U researchmate -d researchmate -F c --no-owner" \
  > /tmp/researchmate.dump

docker compose -f docker-compose.dev.yml up -d postgres
docker cp /tmp/researchmate.dump researchmate_dev_postgres:/tmp/dump.bin
docker exec researchmate_dev_postgres pg_restore \
  -U researchmate -d researchmate \
  --no-owner --role=researchmate --clean --if-exists \
  /tmp/dump.bin
```

**Route B — pull from latest R2 backup** (if SSH not available):

```bash
rclone copy r2:researchmate-backups/latest.dump /tmp/
docker compose -f docker-compose.dev.yml up -d postgres
docker cp /tmp/latest.dump researchmate_dev_postgres:/tmp/dump.bin
docker exec researchmate_dev_postgres pg_restore \
  -U researchmate -d researchmate \
  --no-owner --role=researchmate --clean --if-exists \
  /tmp/dump.bin
```

Route B requires the contributor to have R2 credentials configured locally (see `INFRA.md §6`).

### 5.3 If your local user/db names don't match prod

If you kept `ri` / `research_intelligence` from before, the restore needs an extra remapping. The cleanest fix is to **stop using the old names** and rebuild the local volume:

```bash
docker compose -f docker-compose.dev.yml down
docker volume rm researchmate-dev_pg_data       # destroys local data
docker compose -f docker-compose.dev.yml up -d
# ...then re-import with the new names matching prod
```

If for some reason you must keep old names, use `pg_restore --no-owner --role=ri` and edit your local `.env` to match — but expect occasional friction.

---

## 6. Daily workflow

### Bring stack up

```bash
docker compose -f docker-compose.dev.yml up -d
```

### Tail logs

```bash
docker compose -f docker-compose.dev.yml logs -f api worker
```

### Restart one service after a code change

With `--reload` flag on uvicorn (in the compose file), code changes auto-reload. No restart needed for Python code.

For other changes (e.g., new env vars, dependency installs):

```bash
docker compose -f docker-compose.dev.yml up -d --no-deps --build api worker
```

### Connect to Postgres from host

```bash
psql -h localhost -p 5432 -U researchmate -d researchmate
# password from .env
```

Or from a GUI tool (TablePlus, DataGrip, DBeaver): host `localhost`, port `5432`, user `researchmate`, password from `.env`, database `researchmate`.

### Connect to Redis from host

```bash
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD"
```

### Stop everything

```bash
docker compose -f docker-compose.dev.yml down       # keeps data
docker compose -f docker-compose.dev.yml down -v    # NUKES data — local equivalent of disaster
```

---

## 7. Running migrations locally

Alembic runs from the host (because that's where you author migrations) but talks to the containerized DB on `127.0.0.1:5432`:

```bash
# Apply all pending migrations
alembic upgrade head

# Generate a new migration after editing models
alembic revision --autogenerate -m "add subdomain table"

# Inspect history
alembic history
```

**Connection string** is read from `.env` via `alembic.ini` or `env.py`. Make sure `DATABASE_URL` (or whatever the project uses) points at `localhost:5432`, not `postgres:5432`. The latter only resolves inside the Docker network.

A pragmatic pattern: have `env.py` rewrite `postgres` → `localhost` when it detects it's running outside Docker.

---

## 8. Running pipelines (Layer 0/1/2/3)

Pipelines stay on the contributor's machine, NOT on prod. Prod only hosts the API + worker. Pipeline runs talk to the local Postgres, then any new data is pushed to prod via `pg_dump`/`pg_restore` (current model — see `INFRA.md §13`).

To run a Layer 1 backfill locally:

```bash
docker compose -f docker-compose.dev.yml exec api python -m src.scripts.layer1_retag --limit 10
```

(Adjust the script invocation to match your repo.)

---

## 9. Local auth — bypassing Google OAuth

In production, every API call requires a valid JWT cookie obtained from `/auth/google/callback` after successful Google OAuth. For local dev this is friction. The bypass:

In FastAPI's auth dependency, when `DEV_AUTH_BYPASS=true`:

```python
async def current_user(...) -> User:
    if settings.DEV_AUTH_BYPASS:
        email = request.headers.get("X-Dev-User")
        if email:
            return await get_or_create_user(email=email)
    # ...real OAuth + JWT logic below
```

Then in dev you can curl as anyone:

```bash
curl -H "X-Dev-User: mahdi.ms86@gmail.com" http://localhost:8000/me
```

**This codepath must be guarded by `if settings.ENV == "development"`** in addition to the env flag, so an accidental misconfiguration in prod can't activate it.

---

## 10. Frontend dev

The React mockup at `src/ui/` currently uses hand-written mocks in `src/ui/data.jsx`. M1b replaces this with `src/ui/api_client.js` that calls `http://localhost:8000/api/...`.

For now, the simplest dev loop:

- Frontend served by some dev server (Vite) on `localhost:5173`
- API on `localhost:8000`
- CORS in FastAPI allows `localhost:5173` (only in `ENV=development`)

A FastAPI middleware:

```python
if settings.ENV == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

In production, frontend is served by FastAPI's static-files mount on the same origin, so no CORS needed (see `INFRA.md §8`).

---

## 11. Differences from prod — the explicit list

Things that are intentionally different in dev. Anyone changing these should know why.

| Concern | Local | Prod |
|---|---|---|
| TLS | None — plain HTTP on `localhost:8000` | Caddy + Let's Encrypt |
| Reverse proxy | None | Caddy |
| Frontend origin | `localhost:5173` (Vite) | Same origin as API |
| CORS | Allowed for `localhost:5173` | Disallowed (same-origin) |
| Auth bypass | `X-Dev-User` header accepted | Real Google OAuth only |
| Postgres port | Exposed on `127.0.0.1:5432` | Internal Docker network only |
| Redis port | Exposed on `127.0.0.1:6379` | Internal Docker network only |
| Source code | Bind-mounted into containers | Baked into image at build time |
| Uvicorn `--reload` | On | Off |
| Sentry | Optional (no DSN by default) | Required |
| Backups | None — losing local data is fine | Nightly to R2 |
| Domain / DNS | n/a — all `localhost` | `app.researchmate.app` |
| Log driver | stdout to docker | json-file rotated 10m × 3 |

Everything **not** in this table should match prod. If you're tempted to introduce new divergence, document it here first.

---

## 12. Common gotchas

1. **`postgres` vs `localhost` confusion.** Inside containers, the Postgres host is `postgres`. From your host shell (psql, alembic, IDE plugins), it's `localhost`. Get this wrong and you'll waste 30 minutes wondering why connection refused.

2. **Mixed compose project names.** If you ran `docker compose up` (without `-f`), Docker uses the directory name as the project. If you then run `docker compose -f docker-compose.dev.yml up`, it uses `researchmate-dev`. Volumes don't carry over. Stick to **always** using `-f docker-compose.dev.yml` for dev.

3. **`.env` not picked up.** Compose only auto-loads `.env` from the same directory as the compose file. If you split them, use `--env-file ./path/to/.env` explicitly.

4. **Port 5432 already in use.** Mac users sometimes have a system-installed Postgres running. Either stop it (`brew services stop postgresql`) or change the host-side port mapping in `docker-compose.dev.yml` to `15432:5432` and update `.env` accordingly.

5. **`--reload` doesn't pick up `.env` changes.** Restart the container after changing env vars.

6. **Forgot to `chmod 600 .env`.** Less critical locally than in prod, but still good practice — some tools refuse to load world-readable secrets.

7. **Live-reload mounts on macOS are slow.** First request after a code change might take 1-2s on Apple Silicon Macs. Tolerable; if it becomes painful, add a `:cached` flag to the mount or use Docker Desktop's VirtioFS.

8. **`docker compose down -v` is the local equivalent of disaster.** Wipes named volumes including Postgres data. There is no R2 safety net for local. Have a fresh dump on hand before doing this.

---

## 13. Validating "matches prod"

A simple smoke test: with the dev stack up, confirm:

```bash
docker compose -f docker-compose.dev.yml exec postgres \
  psql -U researchmate -d researchmate -c "\dt"
```

Should list the same 10 tables documented in `INFRA.md §9`. Same row counts (give or take 2 papers due to in-flight Layer 1).

```bash
docker compose -f docker-compose.dev.yml exec postgres \
  psql -U researchmate -d researchmate -c "SELECT extname FROM pg_extension;"
```

Must include `vector`. If not, run `CREATE EXTENSION vector;`.

```bash
docker compose -f docker-compose.dev.yml exec redis redis-cli -a "$REDIS_PASSWORD" ping
```

Must return `PONG`.

If all three pass, your local stack matches prod's data and infra topology.

---

## 14. When ready to deploy a change to prod

Read `INFRA.md §13`. Short version:

```bash
ssh mahdi@5.78.205.155
cd /opt/researchmate
git pull
docker compose up -d --no-deps --build api worker
```

(That assumes the repo is checked out to `/opt/researchmate/`. Currently it isn't — the box has only `docker-compose.yml`, `.env`, `Caddyfile`, `scripts/`, etc., not the application repo. M1b's first deploy will need to reconcile this — likely by cloning the repo to the box or wiring up an image-based deploy via GHCR.)

---

**End of document.**