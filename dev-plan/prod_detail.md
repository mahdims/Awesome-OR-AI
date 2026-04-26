
# researchmate — Production Infrastructure Reference

**Status as of:** April 26, 2026
**Operator:** Mahdi (mahdi.ms86@gmail.com)
**Domain:** researchmate.app, app.researchmate.app
**Public IP (v4):** 5.78.205.155

---

## 1. Hosting & physical layout

### Provider: Hetzner Cloud

- **Account:** registered to Mahdi's personal email
- **Project:** default project, console at https://console.hetzner.cloud/
- **Server name (Hetzner UI):** `ubuntu-2gb-hil-1` — kept as Hetzner's default; not the OS hostname
- **Server type:** **CPX11** (AMD EPYC, 2 vCPU, 2 GB RAM, 40 GB NVMe local disk, 20 TB outbound traffic/month)
- **Location:** **Hillsboro, Oregon, USA (hil1)**
  - Chosen for ~20 ms latency from Vancouver, where the operator is based
  - Note: not in EU, not in Canada — relevant if data residency ever becomes a constraint
- **Cost:** $5.99/month, billed hourly in arrears
- **Backups:** **disabled** at Hetzner level (snapshots not enabled)
  - Disaster recovery is handled at the application layer via R2 (see §10)
- **Floating IP:** none (single-server deployment)
- **SSH key:** added at server creation time, no root password set

### Operating system

- **Distro:** Ubuntu 24.04.4 LTS (Noble Numbat)
- **Kernel:** 6.8.0-110-generic at time of writing; `unattended-upgrades` will roll this forward automatically
- **Hostname (OS):** `researchmate` — set persistent via `/etc/cloud/cloud.cfg.d/99-researchmate.cfg` containing `preserve_hostname: true`
- **Timezone:** `Etc/UTC` (functionally UTC). All logs, cron, and timestamps are in UTC.
- **Swap:** 2 GB at `/swapfile`, `vm.swappiness=10`, persisted in `/etc/fstab`

---

## 2. Users & SSH

### Login model

- **Root SSH:** **disabled** (`PermitRootLogin no`)
- **Password auth:** **disabled** (`PasswordAuthentication no`, `KbdInteractiveAuthentication no`)
- **Sole SSH user:** `mahdi`, member of `sudo` and `docker` groups
- **Auth method:** SSH public key only

### How to log in

```bash
ssh mahdi@5.78.205.155
```

The deploy user's authorized public keys are at `/home/mahdi/.ssh/authorized_keys`. If a new contributor needs access:

1. Have them generate `ssh-keygen -t ed25519` on their machine
2. Send the `.pub` file
3. Append it as a single line to `/home/mahdi/.ssh/authorized_keys` on the server (alternative: create a new user — more isolation, more setup)

### SSH server config

Hardening overrides live at `/etc/ssh/sshd_config.d/99-hardening.conf`:

```
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
X11Forwarding no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
```

---

## 3. Network security

### UFW firewall

Active. Default: `deny incoming`, `allow outgoing`. Inbound allow-list:

| Port | Proto | Purpose |
|---|---|---|
| 22 | TCP | SSH |
| 80 | TCP | HTTP (Caddy uses this for ACME challenges + redirects to HTTPS) |
| 443 | TCP | HTTPS (Caddy) |

All other ports blocked. **Postgres (5432) and Redis (6379) are NOT exposed** — they only exist on Docker's internal bridge network.

### fail2ban

Active. Single jail: `sshd`. Bans for 1 hour after 5 failed auth attempts within 10 minutes. Backend is `systemd` (reads journald, not log files). Config at `/etc/fail2ban/jail.local`.

### Cloud-side firewall (Hetzner)

Not configured. Defense in depth could add rules at Hetzner Cloud Firewall layer; currently only the on-box ufw is in play. Functionally equivalent for a single server.

---

## 4. Auto-patching

`unattended-upgrades` is active and configured for auto-reboot when needed:

- File: `/etc/apt/apt.conf.d/52unattended-upgrades-local`
- Settings: `Automatic-Reboot "true"`, `Automatic-Reboot-Time "04:00"` (UTC), removes unused kernels and dependencies
- Effective behavior: security patches apply nightly. If a kernel update requires a reboot, it happens at 04:00 UTC (= 21:00 Pacific the previous day)

Expected downtime from this: ~30–60 seconds per kernel update, roughly once a month or less.

---

## 5. Domain & DNS

### Registration

- **Registrar:** Cloudflare Registrar (at-cost pricing)
- **Domain:** `researchmate.app`
- **Renewal:** ~$14/year, auto-renews
- **Cloudflare account:** Mahdi's personal email

### DNS records (all on Cloudflare DNS)

| Type | Name | Target | Proxy mode | Purpose |
|---|---|---|---|---|
| A | @ | 5.78.205.155 | DNS only (gray) | Apex; redirects to `app.` |
| A | app | 5.78.205.155 | DNS only (gray) | Main app |
| A | * | 5.78.205.155 | DNS only (gray) | Wildcard for future subdomains |

**Important:** all records are **DNS only**, NOT proxied through Cloudflare. The orange-cloud proxy mode is intentionally disabled so Caddy on the server can issue Let's Encrypt certificates directly via HTTP-01 challenge. If a future contributor enables proxy mode, TLS issuance will break unless Cloudflare Origin certs or Strict mode is also configured.

---

## 6. Cloudflare R2 (object storage)

- **Bucket:** `researchmate-backups`
- **Location:** automatic (Cloudflare picks closest; effectively WNAM)
- **Access pattern:** S3-compatible API
- **API token:** scoped to this bucket only ("Object Read & Write"); cannot list buckets at account level
- **Endpoint:** `https://<account-id>.r2.cloudflarestorage.com` (account ID stored in operator's password manager)
- **Storage class:** Standard
- **Free tier covers our usage** by ~1000x; will not need a paid plan

### rclone configuration

The deploy user's rclone config at `~/.config/rclone/rclone.conf`:

```ini
[r2]
type = s3
provider = Cloudflare
access_key_id = <REDACTED>
secret_access_key = <REDACTED>
endpoint = https://<account-id>.r2.cloudflarestorage.com
acl = private
no_check_bucket = true
```

The `no_check_bucket = true` is required because the bucket-scoped API token can't perform `ListBuckets` or `CreateBucket`. Without this flag, rclone tries to verify/create the bucket on every operation and fails with 403.

### Bucket layout

```
researchmate-backups/
├── daily/                  # rolling 14 days
├── weekly/                 # rolling ~5 weeks (promoted from daily on Mondays)
├── monthly/                # rolling ~6 months (promoted from daily on the 2nd of each month)
└── latest.dump             # always = most recent backup, fixed name for easy retrieval
```

---

## 7. Docker setup

### Engine

- **Source:** Docker's official apt repository (NOT Ubuntu's `docker.io` package)
- **Components:** `docker-ce`, `docker-ce-cli`, `containerd.io`, `docker-buildx-plugin`, `docker-compose-plugin`
- **Compose CLI:** `docker compose` (v2 plugin), NOT the legacy `docker-compose` standalone
- **`mahdi` user is in the `docker` group** — no `sudo` needed for docker commands

### Daemon configuration

`/etc/docker/daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-address-pools": [
    {"base": "172.20.0.0/16", "size": 24}
  ]
}
```

**Why this matters:** without log rotation, Docker's default JSON log driver fills disk indefinitely. The 10m × 3 = 30 MB cap per container is sufficient for our scale.

The `172.20.0.0/16` address pool was chosen to pre-empt subnet conflicts (default `172.17.0.0/16` would also work; this is documented intent for future projects on the same box).

---

## 8. Application stack — `/opt/researchmate/`

All app-related files live under `/opt/researchmate/`, owned by `mahdi:mahdi`.

### Directory structure

```
/opt/researchmate/
├── docker-compose.yml         # source of truth for service topology
├── .env                       # secrets — chmod 600 — git-ignored, never committed
├── caddy/
│   └── Caddyfile             # reverse-proxy config
├── data/                      # currently empty; named Docker volumes hold data
├── backups/                   # local pg_dump output (rolling 3 days)
│   ├── researchmate_*.dump
│   └── backup.log            # cron output
├── secrets/                   # for non-.env secrets (NotebookLM session file later)
├── scripts/
│   └── backup.sh             # nightly backup script
└── RECOVERY.md                # disaster-recovery runbook
```

### Docker Compose project

Project name: `researchmate` (set by `name:` directive in compose file). All containers, networks, and volumes are prefixed with this.

### Services currently running

| Container | Image | Network | Ports exposed | Healthcheck |
|---|---|---|---|---|
| `researchmate_postgres` | `pgvector/pgvector:pg16` | internal only | none (5432 internal only) | `pg_isready` |
| `researchmate_redis` | `redis:7-alpine` | internal only | none (6379 internal only) | `redis-cli ping` |
| `researchmate_web` | `nginxdemos/hello:plain-text` | internal only | none (80 internal only) | none — placeholder |
| `researchmate_caddy` | `caddy:2-alpine` | internal + host | 80, 443 (TCP) | none |

**`researchmate_web` is a temporary placeholder.** It returns the nginx-hello "Server address" page. The next contributor (M1b work) replaces this with the real FastAPI container.

### Network

- One Docker bridge network: `researchmate_internal` (named `internal` in the compose file, prefixed by project name)
- All four services on this network; only Caddy publishes ports to the host

### Named volumes

- `researchmate_pg_data` — Postgres data directory
- `researchmate_redis_data` — Redis AOF persistence
- `researchmate_caddy_data` — Caddy state including TLS certificates and renewal data
- `researchmate_caddy_config` — Caddy runtime config

**Data persistence:** these volumes survive `docker compose down`. They are deleted by `docker compose down -v` (which is destructive — never run on prod).

### Environment variables (`.env`)

Required keys:

```
DOMAIN=app.researchmate.app
ACME_EMAIL=mahdi.ms86@gmail.com

POSTGRES_USER=researchmate
POSTGRES_PASSWORD=<32-char random>
POSTGRES_DB=researchmate

REDIS_PASSWORD=<32-char random>
JWT_SECRET=<64-char random>

ENV=production
LOG_LEVEL=info

SENTRY_DSN=https://<...>@<...>.ingest.sentry.io/<...>
```

The `.env` file is `chmod 600`, owned by `mahdi`, lives at `/opt/researchmate/.env`, and is NOT in any git repo. **Backed up only in operator's password manager.** Loss of `.env` means re-issuing all credentials.

### Caddyfile

`/opt/researchmate/caddy/Caddyfile` (mounted read-only):

```caddy
{
    email {$ACME_EMAIL}
}

{$DOMAIN} {
    encode gzip zstd

    handle /health {
        respond "ok" 200 {
            close
        }
    }

    handle {
        reverse_proxy web:80
    }

    log {
        output stdout
        format console
        level INFO
    }
}

researchmate.app {
    redir https://app.researchmate.app{uri} permanent
}
```

The `/health` endpoint is intentionally separate from the upstream app — it returns 200 even if Postgres or the app is broken. The "deeper" health check (DB reachable, Redis reachable, NotebookLM circuit closed) belongs in FastAPI as `/health/deep` once that exists.

When replacing the placeholder with the real app, the `reverse_proxy web:80` line stays as-is **if the new container is also named `web`** (Compose service name). If renamed (e.g., `api`), update both `docker-compose.yml` and the Caddyfile target.

To reload Caddy without dropping connections:

```bash
docker compose exec caddy caddy reload --config /etc/caddy/Caddyfile
```

`caddy fmt` cannot be run from inside the container (Caddyfile mount is read-only). Use a host-side one-shot:

```bash
docker run --rm -v /opt/researchmate/caddy:/work caddy:2-alpine caddy fmt --overwrite /work/Caddyfile
```

---

## 9. Database state

### Postgres details

- Engine: Postgres 16, vendored as `pgvector/pgvector:pg16` (includes pgvector extension)
- DB name: `researchmate`
- DB user: `researchmate` (owns all tables)
- Connection string from inside Docker network: `postgresql://researchmate:<password>@postgres:5432/researchmate`
- Connection from host: NOT POSSIBLE without `docker exec` or temporarily exposing the port

### Schema state

Migrated from local dev via `pg_dump -F c` → `pg_restore --no-owner --role=researchmate` on April 26, 2026. **Alembic version (revision id):** check via:

```bash
docker exec researchmate_postgres psql -U researchmate -d researchmate -c "SELECT version_num FROM alembic_version;"
```

Tables present (10):

| Table | Purpose |
|---|---|
| `papers` | 702 rows — Layer 0 fetched papers |
| `paper_analyses` | 700 rows — Layer 1 4-agent analysis output |
| `citations` | citation graph data |
| `citation_fetch_log` | bookkeeping for Layer 0 |
| `cocitation_edges` | Layer 2 |
| `research_fronts` | Layer 2 (NOT surfaced in UI per v3 plan) |
| `front_lineage` | Layer 2 |
| `bridge_papers` | Layer 2 |
| `review_updates` | Layer 3 |
| `alembic_version` | migration state |

**The 2-paper gap** between `papers` (702) and `paper_analyses` (700) is normal — it means the Layer 1 pipeline hasn't yet processed two recently fetched papers.

### pgvector

The `vector` extension should be loaded. Verify with:

```bash
docker exec researchmate_postgres psql -U researchmate -d researchmate -c "SELECT extname, extversion FROM pg_extension;"
```

If `vector` is not in the result, `CREATE EXTENSION vector;` is required before any embedding work.

---

## 10. Backups & disaster recovery

### Backup script

`/opt/researchmate/scripts/backup.sh` — runs nightly at 03:30 UTC via `mahdi`'s crontab.

What it does:

1. `pg_dump -F c` from inside Postgres container, written to `/opt/researchmate/backups/researchmate_<UTC-iso>.dump`
2. `rclone copyto` to `r2:researchmate-backups/daily/<filename>`
3. Same upload also overwrites `r2:researchmate-backups/latest.dump` (always = most recent)
4. Local cleanup: deletes local dumps older than 3 days
5. R2 cleanup:
   - `daily/` — files older than 14 days deleted
   - On Mondays, yesterday's daily promoted to `weekly/`; weekly files >35d deleted
   - On the 2nd of each month, the 1st's daily promoted to `monthly/`; monthly files >190d deleted

Logs append to `/opt/researchmate/backups/backup.log`.

### Cron entry

```
30 3 * * * /opt/researchmate/scripts/backup.sh >> /opt/researchmate/backups/backup.log 2>&1
```

### Tested restore path

Restore was tested on April 26, 2026 by restoring `latest.dump` into a throwaway database `restore_test` inside the same container, querying `SELECT COUNT(*) FROM papers;` to verify the row count matched production.

### Recovery runbook

`/opt/researchmate/RECOVERY.md` documents the step-by-step restore procedure including:
- Stop the app stack (keep Postgres up)
- Pull `latest.dump` (or specific date) from R2
- `dropdb` → `createdb` → `pg_restore`
- Verify counts
- Restart stack

**Worst-case scenario:** total Hetzner box loss. Recovery: provision new CPX11 in Hillsboro, run setup commands (currently in this document — recommend converting to a script), restore from R2, point DNS at new IP. ETA: 1–2 hours including DNS propagation.

**Failure modes not covered yet:**
- R2 region outage simultaneous with box loss (no offsite backup of backups)
- Cryptolocker-style attack that encrypts backups before they upload — daily files would be safe but `latest.dump` would be poisoned

---

## 11. Monitoring & alerts

### Sentry

- Project: `researchmate-api`
- Platform: Python / FastAPI
- DSN: stored in `.env` as `SENTRY_DSN`
- Free Developer tier — 5K events/month
- Alert delivery: email to operator
- **Currently not receiving any events** because no app code is running yet. Will start receiving events once FastAPI is deployed in M1b.

### Better Stack (uptime)

- Monitor name: `researchmate health`
- URL: `https://app.researchmate.app/health`
- Frequency: 3 minutes
- Expected: HTTP 2xx (the body keyword check is not active on free tier)
- Alert delivery: email to operator
- Status pages: not configured
- On-call escalation: "Do nothing" if not acknowledged (single-operator project)

### What's NOT monitored yet

- Disk usage (you should learn about full disk before backups fail; nothing currently warns)
- Postgres replication lag (irrelevant — single instance)
- Application metrics (response times, error rates) — needs Prometheus or similar; deferred
- Backup success/failure — the cron script logs but doesn't alert. **Recommended addition:** trigger a Better Stack heartbeat at the end of `backup.sh` so a missed nightly run pages.

---

## 12. Current condition (as of April 26, 2026)

### What works end-to-end

- `https://app.researchmate.app/` returns nginx-hello placeholder over HTTP/2 with valid Let's Encrypt cert
- `https://app.researchmate.app/health` returns `ok` (200)
- HTTP redirect: 80 → 443 via Caddy
- Apex redirect: `https://researchmate.app/` → `https://app.researchmate.app/`
- All four containers running and (where applicable) reporting healthy
- 702 papers + 700 analyses queryable via `docker exec researchmate_postgres psql ...`
- Daily 03:30 UTC backup pipeline functional and tested
- Better Stack pings every 3 min, last-known status: up

### What does NOT work yet (on purpose, planned for M1b+)

- No FastAPI application — `/api/*` paths return whatever nginx-hello does (probably 200 with the same placeholder)
- No Google OAuth, no users, no sessions, no JWT
- No frontend deployed — React mockup (`src/ui/`) is still pure mock data on the operator's laptop
- No structured pipeline running on the box — Layer 0/1 still runs locally and data gets pushed up via `pg_dump`/`pg_restore`. There is no scheduled pipeline on the box yet.
- No NotebookLM integration — `notebooklm-py` not installed, no Google session file present
- No arq job queue — Redis is up, but no worker is consuming jobs
- No subdomain taxonomy, no embeddings, no novelty service, no gap detector — these are M1c/M2/M3/M4 deliverables

### Gotchas to know about before changing anything

1. **Do not run `docker compose down -v`** — the `-v` flag deletes named volumes (including `pg_data`). This is destructive and irreversible without restoring from R2.

2. **Do not `apt install docker.io` or `apt remove docker-ce`** — these would conflict with the Docker CE installation. Update via `apt upgrade` (already auto-handled by unattended-upgrades).

3. **Do not flip Cloudflare DNS to "Proxied" (orange cloud)** without also reconfiguring Caddy for Origin certs or Strict TLS. Doing so will break TLS within minutes.

4. **Do not commit `.env` to git.** It contains DB passwords, JWT secret, Sentry DSN. Add `.env` to `.gitignore` if it isn't already.

5. **rclone's `no_check_bucket = true` config line is required.** If a contributor "cleans up" by removing it, every backup operation will fail with 403.

6. **Caddyfile is mounted read-only.** Run `caddy fmt` from a host-side one-shot, not from inside the running container.

7. **The Postgres user `researchmate` owns all tables.** When restoring dumps from elsewhere (e.g., the operator's laptop, where the user is `ri`), use `pg_restore --no-owner --role=researchmate` to remap.

8. **Caddy cert renewal** happens automatically via Let's Encrypt every ~60 days. Stored in `caddy_data` volume. If that volume is ever deleted, Caddy will re-issue from scratch on next boot — but Let's Encrypt has a rate limit of 50 certs/week per registered domain, so don't make a habit of it.

---

## 13. How to deploy a code change (current model)

This is the operating model from M1b onward.

1. **Develop on laptop** against a local Docker stack mirroring prod (Postgres + Redis at minimum)
2. **Test locally** until satisfied
3. **Build container image** locally OR push code via git and build on the server
4. **SSH to prod, update `docker-compose.yml` and/or rebuild**
5. **Reload/restart only the changed service:** `docker compose up -d --no-deps <service-name>`

**Currently undecided:** whether to use a registry (GHCR) or build on the server. Recommendation: build on the server for simplicity until CI is justified.

---

## 14. Total monthly cost

| Item | Cost |
|---|---|
| Hetzner CPX11 | $5.99 |
| Domain (researchmate.app, amortized) | ~$1.17 |
| Cloudflare DNS, R2, registrar service | $0 |
| Sentry (Developer tier) | $0 |
| Better Stack (free tier) | $0 |
| Gemini API (when M1b+ work begins) | ~$6 |
| NotebookLM (free Google account) | $0 |
| **Total before app code** | **~$7** |
| **Total at full M5 deployment** | **~$13** |

---

## 15. Things a future contributor should ask the operator

If you are a contributor onboarding to this project and you've read this doc but still need:

- **The R2 API key + endpoint** — operator's password manager
- **Sentry DSN** — already in `.env` on the box; can be regenerated from Sentry dashboard
- **Production `.env` content** — operator's password manager
- **NotebookLM Google account credentials** — does not exist yet (Phase 9 deferred)
- **Hetzner login** — operator's account; for shared admin, ask operator to add you to the project

---
