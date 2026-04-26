# Infrastructure & Serving Plan

Consolidated list of every infrastructure choice for the Research Intelligence project. Grouped by category, with the decision first and the rationale/alternative in a line. Pairs with [backend-plan.md](backend-plan.md).

## Hosting & compute

- **VPS provider: Hetzner Cloud** — best price-to-power ratio, already wanted a VPS for other projects. Alternatives considered: Fly.io (more managed, ~$30/mo), Vultr/DO Toronto (slightly more expensive).
- **Region: Hillsboro (hil1)** — 15-25ms from Vancouver. Alternatives considered: Helsinki (would be ~170ms, rejected), Ashburn (fine but farther).
- **Instance: CX32** (4 vCPU, 8GB RAM, 80GB NVMe, ~$8/mo) — headroom for other projects on the same box. Alternative: CX22 (~$5/mo) — sufficient for just the research tool alone.
- **Backups: Hetzner automatic snapshots enabled** — +20% surcharge, ~$2/mo, weekly automated.
- **OS: Ubuntu 24.04 LTS** — supported until 2029, long stable window.

## Database

- **Engine: Postgres 16 + pgvector extension** — pgvector needed for embeddings. Docker image: `pgvector/pgvector:pg16`.
- **Hosting: self-hosted in Docker on the VPS** — bundled with everything else. Alternative considered: Neon free tier (would've saved ~$0 net since already on Hetzner).
- **Access: internal Docker network only** — no public port published. Connections from app/worker only.
- **Backups: nightly `pg_dump` to R2** via cron + rclone, 14-day rolling retention. Hetzner snapshots are the "oops broke the box" layer; pg_dumps are the "oops dropped a table" layer.

## Cache / queue

- **Redis 7-alpine** — self-hosted in Docker, internal network only. Used for arq job queue + Redis-backed rate-limit buckets.
- Alternative considered: Upstash (paid, rejected since Redis colocated on the VPS is simpler and free).

## Object storage

- **Cloudflare R2** — used for (a) cached NotebookLM audio/video MP3s, (b) offsite Postgres backups. Free tier covers the volume. Zero egress fees — critical for audio playback.
- Alternatives considered: S3 (rejected: $0.09/GB egress), Backblaze B2 (fine but R2 is cheaper and more mature).

## DNS & TLS

- **DNS: Cloudflare** — free, fast, good API. Orange-cloud *off* for the app hostname since Caddy handles TLS directly. Orange-cloud *on* fine for other static things.
- **Domain registrar: Cloudflare Registrar** — at-cost pricing (~$12/year). Alternative: keep wherever the domain is now; doesn't matter.
- **TLS: Caddy 2 with automatic Let's Encrypt** — one-line config per hostname. Alternative considered: nginx + certbot (more config, no reason to pick it over Caddy in 2026).

## Reverse proxy / app routing

- **Caddy 2** — serves TLS + routes hostname → container. Same tool for multiple projects on the same box (research project, future side projects) by adding hostname blocks.

## Application framework

- **Backend: FastAPI** — already chosen in the backend plan. Async-native, Pydantic, auto OpenAPI for generating frontend TypeScript types.
- **Frontend: existing React mock adapted to fetch from `/api`** — no rewrite, just replace the mock `data.jsx` with a real API client.
- **Origin: single-origin deployment** — FastAPI serves the static UI itself. Eliminates CORS/CSRF complications entirely.

## Background jobs

- **Queue framework: arq** (async Redis queue). Chosen over RQ because notebooklm-py is async-native; running async code inside sync RQ workers is brittle.
- **Scheduler: arq cron tasks + Postgres advisory locks** — guards against duplicate fires on restart/multi-instance.
- **Retries: 3 with exponential backoff (60s, 5m, 30m), then dead-letter** — job state mirrored to Postgres `jobs` table for UI polling.

## Authentication

- **Google OAuth 2.0 via Authlib + JWT in HttpOnly SameSite=Lax cookie** — rolled in-house, ~200 lines. Alternatives considered: Auth0/Clerk (overkill), Supabase Auth (only if using Supabase DB).
- **Access control: explicit email allow-list in env var** — not a domain check. Since this is personal not university-bound, flexibility to add friends outside any single domain.

## LLM / AI services

- **Language model: Google Gemini (Flash 2.x for extraction, embedding-004 for vectors)** — cheapest option at this volume, same provider already in use. Hard monthly billing cap set at $50. Alternatives considered: Claude Haiku (4-5× price), OpenAI (~similar to Flash), local Ollama (rejected — quality too low for extraction).
- **Embedding model: `text-embedding-004` at 768 dims** — committed decision with a documented upgrade path.
- **NotebookLM: free personal Google account via notebooklm-py** — for notebooks, audio overviews, video, mind maps, chat. Risk accepted: auth breaks every 30-90 days, manual re-login drill.
- **Dedicated Google account for NotebookLM automation** — not the main Gmail, not any institutional email. Fresh account whose only purpose is this. Isolates the "account gets flagged" risk.

## Email

- **Transactional: Gmail SMTP from a project Gmail** — free, ~500/day limit, sufficient for a small team. Alternatives for later: Resend ($0 for 3K/month) when more volume or external sends are needed.

## Monitoring & observability

- **Error tracking: Sentry Developer tier** — free up to 5K errors/month.
- **Uptime monitoring: BetterStack or UptimeRobot (external, free tier)** — pings `/health` from outside the network so outages are heard about even when the box is down. Internal Uptime Kuma on the same box is pointless for this.
- **Logs: Docker json-file driver with rotation (`max-size: 10m, max-file: 3`)** — configured in `daemon.json` day one, otherwise logs silently eat disk. Escalate to Axiom free tier if Docker-log grepping becomes frequent.

## Secrets management

- **`.env` file on the server, git-ignored, 600 permissions, owned by deploy user** — simplest workable approach for a solo operator. No Vault, no SOPS, no cloud secret manager. Back it up to a password manager.
- **JWT signing key: generated once, stored in `.env`** — rotate only if leak is suspected.

## Orchestration

- **Docker Compose** — one `docker-compose.yml` per project under `/opt/<project>/`. No Kubernetes, no Swarm, no Nomad. 3-4 independent project stacks on the same CX32 this way.
- **Deploy: `git pull && docker compose up -d --build`** — no CI/CD for v1. Ship from laptop via SSH.

## Security baseline

- **SSH: key-only, root login disabled, password auth disabled, non-root deploy user**
- **Firewall: ufw allowing only 22/80/443** (on the box itself, Hetzner Cloud Firewall as a second layer optional)
- **`fail2ban`** for brute-force protection
- **`unattended-upgrades`** for auto-applying security patches
- **Docker network isolation**: Postgres and Redis only on internal network, never published

## Local development

- **Same `docker-compose.yml`** (or a `docker-compose.dev.yml` overlay) so local and prod environments are identical. No SQLite/Postgres split — that path is a bug factory.
- **Tests: testcontainers-python** for spinning disposable Postgres per pytest session.

## Data migration (SQLite → Postgres)

- **Custom Python migration script** — `src/scripts/migrate_sqlite_to_pg.py`. Not pgloader, because types are also being promoted (TEXT-JSON → JSONB, BLOB → vector).
- **Alembic** from day one for schema migrations. Non-negotiable.

---

## Monthly cost summary

| Item | Cost |
|---|---|
| Hetzner CX32 (Hillsboro) | ~$8 |
| Hetzner backups | ~$2 |
| Cloudflare R2 | $0 (under free tier) |
| Cloudflare DNS | $0 |
| Domain (Cloudflare Registrar) | ~$1 |
| Gemini API | ~$6 |
| NotebookLM | $0 |
| Sentry | $0 |
| Uptime monitor | $0 |
| Email (Gmail SMTP) | $0 |
| **Total** | **~$17/mo** |

One-time: domain registration if not already owned (~$12), and ~a day of initial setup.

---

## Things explicitly *not* chosen (and why)

- **Kubernetes** — catastrophically over-engineered for this.
- **Managed Postgres** (Neon/Supabase/Fly) — no net savings since already paying for a VPS that'd be used anyway.
- **Separate cache/queue VPS** — waste of money at this scale; Redis on the app box is fine.
- **Paid TTS (ElevenLabs/OpenAI TTS)** — only a fallback if NotebookLM breaks long-term; not in v1.
- **CDN in front of the app** — unnecessary for an auth-gated internal tool. Cloudflare DNS is enough.
- **CI/CD pipeline** — manual deploy from laptop is fine for a solo developer. Add GitHub Actions only when the deploy step starts to bug.
- **Full-text search engine** (Meilisearch/Typesense/Elasticsearch) — Postgres FTS + pgvector is sufficient.
- **Separate API gateway** — FastAPI itself is the edge.
- **Kubernetes secrets / Vault / SOPS** — one developer, one `.env` file, done.
- **Sentry paid tier** — upgrade only if 5K errors/month gets exceeded, which means something is badly wrong.
- **Datadog / New Relic** — hilarious overkill.
- **Multi-region deployment** — single Hillsboro box is fine.
- **Shared Postgres for multiple projects** — separate container per project instead, avoids cross-project migration headaches.

---

## Next deliverables (pending)

When ready to execute, the concrete infra artifacts to produce:
- `docker-compose.yml` (app + worker + Postgres + Redis + Caddy)
- `Caddyfile` (TLS + routing)
- Bootstrap shell script for a fresh Ubuntu 24.04 box (user, firewall, Docker, fail2ban, unattended-upgrades)
- Single-page operational runbook (deploy, recover, re-auth NotebookLM, resize disk)
