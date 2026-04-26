# Operator Runbook — researchmate.app

Day-to-day operations for the production box. Pairs with
[/opt/researchmate/RECOVERY.md](https://app.researchmate.app/) on the box itself
(disaster recovery) and [dev-plan/infra-plan.md](../dev-plan/infra-plan.md)
(architecture decisions).

**Box:** `mahdi@5.78.205.155` · **Domain:** `app.researchmate.app` · **Compose root:** `/opt/researchmate/`

---

## One-time laptop setup

1. **SSH alias.** Append to `~/.ssh/config`:
   ```
   Host researchmate
       HostName 5.78.205.155
       User mahdi
       IdentityFile ~/.ssh/id_ed25519
       ServerAliveInterval 60
   ```
   After this, `ssh researchmate` works. The scripts here use `mahdi@5.78.205.155` directly via `.env.prod` — both forms are fine.

2. **`.env.prod` for the push scripts.**
   ```bash
   cp scripts/prod/.env.prod.example scripts/prod/.env.prod
   ```
   The default values match the box. Edit only if `LOCAL_DATABASE_URL` differs from `postgresql+psycopg://ri:ri_dev@127.0.0.1:5432/research_intelligence`.

   `.env.prod` is gitignored. **It contains no secrets** — real DB credentials live on the box at `/opt/researchmate/.env` (chmod 600). Losing this laptop file is harmless; recreate from the example.

3. **`pg_dump` / `psql` on the laptop** (for `prod-data-push.sh` only):
   ```bash
   brew install libpq && brew link --force libpq
   ```

---

## One-time box setup (M1b prerequisite — not needed yet)

Once M1b is ready to ship, clone the repo onto the box so `prod-migrate.sh` and `prod-deploy-app.sh` have a working tree:
```bash
ssh mahdi@5.78.205.155 'git clone https://github.com/mahdims/Awesome-OR-AI.git /opt/researchmate/app'
```

For M1a today this is **not required** — the only migration (`0001_initial_schema`) was applied to prod via the 2026-04-26 `pg_restore`, and no app code runs on the box.

---

## Readiness check (run before any prod write)

```bash
# 1. SSH + box identity
ssh mahdi@5.78.205.155 'whoami && uname -a'

# 2. All 4 containers up + healthy
ssh mahdi@5.78.205.155 'docker compose -f /opt/researchmate/docker-compose.yml ps'

# 3. TLS + apex redirect
curl -sI https://app.researchmate.app/health         # expect 200
curl -sI https://researchmate.app/                   # expect 301 → app.

# 4. Alembic version on prod matches local
ssh mahdi@5.78.205.155 'docker exec researchmate_postgres psql -U researchmate -d researchmate -tAc "SELECT version_num FROM alembic_version;"'
# expect: 0001_initial_schema

# 5. pgvector loaded
ssh mahdi@5.78.205.155 'docker exec researchmate_postgres psql -U researchmate -d researchmate -tAc "SELECT extversion FROM pg_extension WHERE extname='\''vector'\'';"'

# 6. Row counts (today: 702 / 700)
ssh mahdi@5.78.205.155 'docker exec researchmate_postgres psql -U researchmate -d researchmate -c "SELECT (SELECT count(*) FROM papers) AS papers, (SELECT count(*) FROM paper_analyses) AS analyses;"'

# 7. Last backup in last 24h
ssh mahdi@5.78.205.155 'tail -3 /opt/researchmate/backups/backup.log && ls -lt /opt/researchmate/backups/*.dump | head -3'

# 8. Better Stack dashboard — last 24h all green.
```

If any check fails, **do not run the push scripts**. Investigate first.

---

## The push scripts

All three live under `scripts/prod/`. Run from the repo root on the laptop. Each supports `--dry-run` which prints what would happen without contacting prod (or, for `prod-migrate.sh`, runs `alembic current` instead of `upgrade`).

### `prod-migrate.sh`

Apply pending Alembic migrations to prod Postgres.

- **When:** after committing a new migration locally and verifying it works against your local DB.
- **How it works:** SSHes to the box, launches a one-shot `python:3.13-slim` container on the `researchmate_internal` Docker network, mounts `/opt/researchmate/app` (the cloned repo), reads creds from `/opt/researchmate/.env`, runs `alembic upgrade head`. Exits cleanly. No host-side Python install on the box.
- **Today (M1a):** repo isn't on the box yet, so the script exits with a clear "no migrations dir" message. Once you do the one-time clone (M1b prereq), `--dry-run` will report `0001_initial_schema (head)` and a real run will be a no-op (already at head).

### `prod-data-push.sh`

Replace prod's database with a fresh dump from the laptop. Destructive.

- **When:** after a local pipeline run produces new papers/analyses and you want prod to mirror local. Until M1b lands, this is the *only* way new data reaches prod.
- **Safety:**
  - Type `RESTORE` to confirm.
  - Aborts if local has fewer `paper_analyses` rows than prod.
  - The previous prod state is preserved in last night's R2 backup (`r2:researchmate-backups/latest.dump`).
- **Flow:** `pg_dump -F c` on laptop → `scp` to box → `dropdb` + `createdb` + `pg_restore` inside the postgres container. The placeholder `web` service is stopped during restore (currently a no-op since it's just nginx-hello).

### `prod-deploy-app.sh`

Deploy the FastAPI app. Stub during M1a.

- **Today (M1a):** detects that no `api` service exists in prod's compose file and exits 0 with a deferred-stub message.
- **M1b+:** SSHes to the box, runs `git pull` in `/opt/researchmate/app`, `docker compose build api`, applies any pending Alembic migrations (`compose run --rm api alembic upgrade head`), restarts the `api` service (`up -d --no-deps api`), curls `/health/deep` to verify.
- **Build-on-server, no registry** (`infra-plan.md` left this open; this runbook is the place we committed). Switch to GHCR later by adding one `image:` line to compose and replacing `build api` with `pull api`.

---

## Rollback

### Bad migration
1. Stop the app: `ssh mahdi@5.78.205.155 'docker compose -f /opt/researchmate/docker-compose.yml stop api'` (M1b+).
2. Either downgrade: `bash scripts/prod/prod-migrate.sh` after editing the script's `ALEMBIC_CMD` to `downgrade -1`, or restore from backup (next section).

### Bad data push / corrupted DB

Use the box's recovery runbook at `/opt/researchmate/RECOVERY.md`. Summary:
1. SSH to the box.
2. `rclone copy r2:researchmate-backups/latest.dump /opt/researchmate/backups/` (or pick a specific `daily/`/`weekly/`/`monthly/` file).
3. Stop the app, drop+create the DB, `pg_restore --no-owner --role=researchmate`, restart.

ETA: ~5 minutes.

### Lost laptop / `.env.prod`

There's no laptop secret. Re-clone the repo, `cp scripts/prod/.env.prod.example scripts/prod/.env.prod`, done. Real DB password lives in the box's `/opt/researchmate/.env` and the operator's password manager.

### Total box loss

See `/opt/researchmate/RECOVERY.md`. Provision new CPX11 in Hillsboro, run setup commands (currently in the operator's reference doc — recommend converting to a script in a future "ops polish" PR), restore latest dump from R2, point Cloudflare DNS at the new IP. ETA: 1–2 hours including DNS propagation.

---

## Known gaps (not blockers for M1a)

- `backup.sh` doesn't ping Better Stack, so a silently failing nightly run goes unnoticed. Add a heartbeat call at the end of the script.
- No disk-fill alarm on the 40 GB box. A `df` check + Better Stack push at the top of `backup.sh` is enough.

Both belong to a small ops-polish PR after M1a, not the SQL dialect sweep.
