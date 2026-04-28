# M1b production cutover — one-time operator steps

After this milestone, prod's stack stops serving `nginxdemos/hello` and starts serving the real FastAPI app. The build still happens on the box (no registry).

Two files on the prod box need a one-time edit. Both stay outside the repo because they hold prod-tuned settings (no bind mount, `COOKIE_SECURE=true`, no host port binding, etc.) — they live alongside `.env` in `/opt/researchmate/`.

## 1. `/opt/researchmate/docker-compose.yml`

Replace the `web` service block with an `api` service that builds from the cloned repo:

```yaml
  api:
    build:
      context: /opt/researchmate/app
    container_name: researchmate_api
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    env_file: /opt/researchmate/.env
    environment:
      DATABASE_URL: postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      REDIS_URL: redis://redis:6379/0
      COOKIE_SECURE: "true"
      ENV: production
```

Remove the existing `web` block (the `nginxdemos/hello` placeholder).

## 2. `/opt/researchmate/caddy/Caddyfile`

Point the reverse proxy at the new service name:

```diff
-        reverse_proxy web:80
+        reverse_proxy api:80
```

## 3. Apply both changes (still on the box)

```bash
cd /opt/researchmate
docker compose down web 2>/dev/null || true   # stop the placeholder if still running
# (no `up` yet — prod-deploy-app.sh from your laptop will build + start the api service)
docker compose exec caddy caddy reload --config /etc/caddy/Caddyfile
```

## 4. Operator prerequisites in `/opt/researchmate/.env`

For the OAuth flow to actually authenticate users:

```bash
GOOGLE_OAUTH_CLIENT_ID=...                              # from Google Cloud Console
GOOGLE_OAUTH_CLIENT_SECRET=...                          # from Google Cloud Console
GOOGLE_OAUTH_REDIRECT_URI=https://app.researchmate.app/auth/google/callback
JWT_SECRET=...                                          # openssl rand -hex 64; keep stable
BOOTSTRAP_ADMIN_EMAILS=mahdi.ms86@gmail.com             # comma-sep allowlist bypass
```

If these aren't set yet, the rest of the app still works — `/auth/google/login` returns 503 with a clear message until the credentials land.

## 5. From your laptop

```bash
bash scripts/prod/prod-migrate.sh        # applies 0003 + 0004 to prod Postgres
bash scripts/prod/prod-deploy-app.sh     # builds api, restarts service, hits /health/deep
```

After that:
- `https://app.researchmate.app/health/deep` returns `{db: ok, redis: ok}`.
- `https://app.researchmate.app/` serves the React UI which calls `/api/init`.
- Visiting `/auth/google/login` redirects to Google → on success, `/api/me` returns your user.

## Rollback

If anything breaks and the placeholder needs to come back temporarily:
```bash
ssh mahdi@5.78.205.155 'cd /opt/researchmate && docker compose stop api'
# revert Caddyfile reverse_proxy target from api:80 → put back any working alternative
```
The previous nightly R2 backup is untouched; data is safe.
