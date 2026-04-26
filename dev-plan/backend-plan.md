# Backend Plan — Research Intelligence UI → Functioning Product (v3)

## Context

The React mockup at [src/ui/](../src/ui/) is a full-fidelity product vision (9 tabs, paper drawer, tweaks panel) driven entirely by hand-written mock data in [src/ui/data.jsx](../src/ui/data.jsx). Most buttons are no-ops. The goal: turn it into a real product for a small `@ku.edu.tr` team deployed to a cloud host, with full feature depth (notebooks, audio/video, novelty check, user-created topics).

**Foundations (keep & build on):**
- **Layer 0** ([src/layer0/fetch_and_score.py](../src/layer0/fetch_and_score.py)) — ArXiv/S2/HF intake + relevance filter
- **Layer 1** ([src/layer1/pipeline.py](../src/layer1/pipeline.py)) — 4-agent deep analysis writing `paper_analyses` (285 papers, 42 columns)
- **Layer 3** ([src/layer3/](../src/layer3/)) — living reviews, email, graph viz

**Dropped from the product surface:**
- **Layer 2 fronts** ([src/layer2/](../src/layer2/)) — computation keeps running for internal email reports, but `research_fronts` / `bridge_papers` / `cocitation_edges` are **not** exposed through the UI. The user's curated/created *subdomains* are the only taxonomy the product shows.

**Key external dependency (verified 2026-04-23):**
- **[notebooklm-py](https://github.com/teng-lin/notebooklm-py)** — unofficial driver for Google NotebookLM.
  - Confirmed surface: notebooks CRUD, sources (PDF/URL/YouTube/Drive/text/audio/video), audio overview, video overview, mind map, chat with history, study guide, briefing doc, quizzes, flashcards, slides, sharing (viewer/editor roles).
  - Async-first API (`async with`, `await`) — **this forces the worker stack below**.
  - Browser-based login (`notebooklm login` opens a browser). Tokens live in a session file; no headless refresh flow. Auth break is a manual ops task.
  - Author's own README recommends it for "prototypes, research, personal projects." We are using it off-label for a team product; the plan owns that risk explicitly (circuit breaker, degrade-not-fail UX, runbook).

**User decisions captured:** small team w/ accounts · cloud · full feature depth · hand-curated subdomains + user-created topics additive on top · layers 0/1/3 are the foundation · layer 2 not surfaced · notebooks via notebooklm-py.

---

## What changed from v2 → v3 (adversarial review)

The v2 plan made several claims that don't survive contact with reality. v3 walks them back:

1. **v2: three auto-detectors ship in v1.** Reality: all three still depend on free-text fields (`extensions.next_steps`, `experiments.benchmarks`) that haven't been normalized yet, and "novel + un-cited" can't distinguish paradigm shifts from papers too new to have citations. **v3: Gaps tab ships as a manual curation workspace** backed by the three-kind typology; auto-suggestions are Phase 2 research.
2. **v2: NotebookLM cost is $0.** Reality: Google rate-limits aggressively (daily audio caps hit real teams), and the auth-break ops burden is real labor. **v3:** cost section names rate limits; runbook section names the auth-break drill.
3. **v2: `vector(768)` hardcoded.** Reality: locking a specific dimension before naming the model is a trap. **v3:** pin Gemini `text-embedding-004` at 768 dims *as the decision*, and document the upgrade path (new column `embedding_v2`, dual-read, cutover).
4. **v2: single `subdomain_id` column.** Reality: correct for the UI contract, but it silently loses the top-3 candidates that the review queue needs and closes the door on "secondary tags" later. **v3:** `paper_subdomains(paper_id, subdomain_id, is_primary, confidence, source)` with a uniqueness constraint on `(paper_id) WHERE is_primary`. UI still reads one row per paper; audit and alternatives survive.
5. **v2: M1b in 3 weeks.** Reality: FastAPI shell + OAuth + invites + JWT cookies + user state + read-only routers + search endpoint + data.jsx rewrite + Fly deploy is not 3 weeks for a single dev. **v3: 4 weeks** (M1b spans weeks 2–5).
6. **v2: "SQL dialect sweep."** Reality: **77 occurrences of `INSERT OR REPLACE` / `json(` / `sqlite3.connect` across 13 files** (measured 2026-04-23). v3 lists the files and sizes M1a as a full week, not a bullet.
7. **v2: tests as a separate section at the end.** Reality: that's where tests go to die. **v3:** each milestone's acceptance criteria include the test that gates it.
8. **v2: novelty pipeline is engineering.** Reality: the verdict-agent prompt is research — getting "breakdown_ratios" that agree with a panel of humans is an evaluation problem. **v3:** M4 is gated on a small held-out verdict-quality evaluation, not just "endpoint returns 200."
9. **v2: chat proxy in M2.** Reality: confirmed to exist in notebooklm-py; keep it, but it's stretch within M2.
10. **v2: three visibility states.** Reality: `curated` + `private` cover the whole team's workflow for v1. **v3:** ship those two; defer `shared` to M5.
11. **v2 missed: UI has hardcoded subdomain IDs in 5 files** ([app.jsx:8](../src/ui/app.jsx), [tabs/today.jsx:9,111](../src/ui/tabs/today.jsx), [tabs/queue.jsx:81](../src/ui/tabs/queue.jsx), [tabs/novelty.jsx:5](../src/ui/tabs/novelty.jsx), [tabs/gaps.jsx:9,10](../src/ui/tabs/gaps.jsx)). These must be parameterized from server data before M3's real taxonomy lands.
12. **v2 missed: YAML seed is a researcher deliverable, not an engineering task.** M3 is blocked on the user producing `research_config/subdomains.yaml`.
13. **v2 missed: transitional rollback story.** **v3:** pg_dump + feature flags + documented "disable feature / fall back to read-only" procedures.

---

## UI-field → DB-column mapping (Paper shape)

| UI field in `window.PAPERS[i]` | DB source | Status |
|---|---|---|
| `id`, `title`, `authors`, `affiliations`, `category`, `date`, `code_url` | `paper_analyses.*` / `papers.*` | ready |
| `priority` | `smart_prioritization.compute_priority_score()` | computed on export |
| `must_read`, `changes_thinking`, `team_discussion`, `reasoning` | `paper_analyses.significance_*` | ready |
| `relevance.{methodological,problem,inspirational}` | `paper_analyses.relevance_*` | ready |
| `brief`, `methods[]`, `problems[]`, `problem_short` | `paper_analyses.*` | ready |
| `benchmarks[]`, `baselines[]`, `vs_baselines{}` | `paper_analyses.experiments_benchmarks`, `results_vs_baselines` | free text — normalized in M3 |
| `confidence_results`, `framework_lineage`, `llm_model`, `novelty_type`, `new_benchmark` | `paper_analyses.methodology_*`, `lineage.*` | ready |
| **`subdomain`** (single id per UI contract) | **does not exist** | new `paper_subdomains` row with `is_primary=true` |
| `window.SUBDOMAINS[sdId].{name,category,tagline,weekly}` | **does not exist** | new `subdomains` table + YAML seed |
| `window.GAPS[sdId][]` (typed gaps) | none | new `gaps` table populated manually in v1 |
| `window.SIGNALS[sdId][]` | none | nightly aggregation over papers+subdomains |
| `sotaFor()`, `benchmarksFor()`, `baselinesFor()`, `labsFor()` | partially derivable | needs canonical-benchmark normalization |
| Queue state, pins, follows, notes | none | per-user state |

---

## Architecture

```
src/
├─ api/                             NEW — FastAPI application
│  ├─ main.py                       app factory + UI static mount (same origin = no CORS)
│  ├─ settings.py                   pydantic-settings
│  ├─ deps.py                       DB session, current_user, rate_limit
│  ├─ auth/                         Google OAuth + JWT cookie + invite list
│  ├─ routers/                      one file per tab
│  ├─ services/
│  │  ├─ notebooks.py               wraps notebooklm-py (async)
│  │  ├─ notebooklm_client.py       health-check + circuit breaker + session file mgmt
│  │  ├─ novelty.py                 embed + vector search + verdict agent
│  │  ├─ search.py                  lexical + semantic
│  │  └─ subdomain_assign.py
│  └─ schemas/                      pydantic response models (mirror UI shapes)
├─ db/
│  ├─ schema.sql                    kept as docs; source of truth is alembic
│  └─ migrations/                   NEW — alembic
├─ layer1/
│  └─ embedder.py                   NEW — populate paper_analyses.embedding via Gemini
├─ layer4/                          NEW — UI-facing feature pipelines
│  ├─ subdomain_assigner.py
│  ├─ benchmark_normalizer.py
│  └─ signal_detector.py
│  (gap_detector.py deferred to Phase 2 — see "Gaps" below)
├─ jobs/                            NEW — background workers
│  ├─ worker.py                     arq worker (async — required by notebooklm-py)
│  ├─ tasks.py                      retry/DLQ configured
│  └─ scheduler.py                  cron with PG advisory locks
└─ ui/                              EXISTING — adapt to fetch from /api
   └─ api_client.js                 NEW — replaces data.jsx
```

### Stack decisions (with rationale)

- **Framework: FastAPI** — Pydantic already in Layer 1, async native, auto OpenAPI for TS-type generation.
- **DB: Postgres + pgvector** self-hosted (see [infra-plan.md](infra-plan.md)). **Local dev uses the same stack via Docker Compose** — no SQLite/Postgres split. One path avoids a subtle-bugs factory.
- **Auth: Google OAuth 2.0 + JWT in HttpOnly SameSite=Lax cookie**, invite-list at callback. CSRF not a concern because SameSite=Lax + same-origin mount. Explicit CORS only for a dev-mode `localhost:5173` origin.
- **Origin model: single origin.** FastAPI serves the UI as static files. No CORS in prod.
- **Background jobs: arq (NOT RQ).** notebooklm-py is async-first; running it under RQ (sync) means every call gets wrapped in `asyncio.run()` or dispatched to an executor — brittle and slow. `arq` is Redis-backed and async-native. Configure: 3 retries with exp backoff (60s, 5m, 30m), dead-letter set, job state mirrored to Postgres `jobs` table so UI polling is reliable.
- **Scheduler: Postgres advisory locks inside a scheduled arq task.** `pg_try_advisory_lock(hash(task_name))` ensures duplicate fires (redeploy overlap, multi-instance) are no-ops.
- **Storage: Cloudflare R2 bucket** for any audio we cache. NotebookLM artifacts live in the user's Google Drive; we store URLs in Postgres plus an R2-cached copy of the MP3 so playback doesn't break when Google rotates URLs.
- **Rate limiting: Redis-backed leaky bucket** per user+endpoint. Defaults: 10 novelty/hour, 5 notebook-generate/hour, 3 audio-generate/hour, 20 search/min. 429 + `Retry-After` on exceed.

---

## NotebookLM integration (the center of gravity)

**Library:** [teng-lin/notebooklm-py](https://github.com/teng-lin/notebooklm-py) — confirmed surface, async API, browser-login auth.

**Operational reality we're accepting:**
- Rate limits apply (team-scale users have hit daily audio caps). Our user-facing rate limit (3 audio-generate/hour/user × 5 users = 15/hour max across the team) stays well under typical limits, but we surface 429s from NotebookLM as "queue is full, retry in 1 hour."
- Auth tokens expire. No programmatic refresh. **Runbook:** the NotebookLM automation account (dedicated Google account, not personal, not institutional) runs `notebooklm login` on the app host, commits the session file to a secret. Monitoring alerts when circuit breaker trips for auth reasons.
- notebooklm-py's own README says it's for prototypes. We are accepting that risk with graceful degradation: when it's down, the UI shows "Notebook/audio features are temporarily unavailable" and every other tab keeps working.

**UI feature → library mapping:**

| UI feature | Calls |
|---|---|
| Notebook tab "Save + generate audio" | `notebook.create_from_papers(...)` → `notebook.generate_audio_overview()` → returns MP3 URL + notebook link |
| Notebook tab "Spawn NotebookLM" button | `notebook.create_from_papers(...)` → return `share_url()` for viewer-role share |
| Today tab "Play all" audio strip | Per-subdomain weekly Audio Overview auto-generated nightly (opt-in) |
| Paper drawer "Generate audio" | Single-paper notebook → audio overview |
| Novelty tab "Spawn notebook" | Notebook populated with the 10 closest prior-work papers + user query as text source |
| Paper drawer chat (stretch within M2) | Proxy into NotebookLM `chat()` API |

**Client architecture** ([src/api/services/notebooklm_client.py](../src/api/services/notebooklm_client.py)):
- Thin async wrapper over notebooklm-py.
- **Circuit breaker** (async breaker): after 3 consecutive failures, short-circuit for 5 min; UI shows "NotebookLM unavailable."
- **Health check:** nightly cron calls `list_notebooks()` (cheap, no-op). Failure → Sentry alert + circuit opens.
- **Idempotency:** artifacts are re-creatable; we store the inputs (paper IDs + title), so a failed generate can be retried without data loss.
- **Cache:** once audio URL is fetched, copy MP3 into R2 and serve from there. Even if Google rotates the URL, our copy keeps playing.

**DB row:** `notebooklm_notebooks(id, nlm_notebook_id, title, query, source_paper_ids JSONB, audio_url, audio_r2_key, video_url, mindmap_url, chat_available, owner_user_id, visibility, created_at)`.

---

## New DB tables (via Alembic migrations)

```sql
users(id, email UNIQUE, name, avatar_url, role, created_at)
invites(email, invited_by_user_id, accepted_at)

-- Subdomains. visibility: 'curated' (YAML seed) | 'private' (single user). 'shared' deferred to M5.
subdomains(
  id TEXT PK,               -- 'evo_llm_search' or 'usr_<uuid>'
  name, tagline, category,
  created_by_user_id NULL,  -- NULL for curated
  visibility TEXT CHECK IN ('curated','private'),
  keywords JSONB,
  canonical_benchmark_ids JSONB,
  canonical_baseline_ids JSONB,
  embedding vector(768),    -- name+tagline+keywords
  created_at
)

-- Per-paper subdomain memberships. is_primary enforces the UI "one primary" contract
-- while preserving top-3 candidates and user-topic overlays.
paper_subdomains(
  paper_id, subdomain_id,
  is_primary BOOLEAN NOT NULL DEFAULT false,
  confidence REAL,          -- NULL for manual adds
  source TEXT,              -- 'auto' | 'review:<user_id>' | 'manual:<user_id>'
  created_at,
  PRIMARY KEY (paper_id, subdomain_id)
);
-- Exactly one primary per paper:
CREATE UNIQUE INDEX paper_subdomains_one_primary
  ON paper_subdomains(paper_id) WHERE is_primary;

-- Low-confidence review queue — references paper_subdomains candidate rows
paper_subdomain_review(
  paper_id PK,
  top_k JSONB,              -- [{subdomain_id, confidence}, ...]
  status TEXT,              -- 'pending' | 'resolved'
  resolved_by_user_id, resolved_at
)

-- Gaps — three kinds only. v1 is manually curated; detectors come later.
-- kind: 'unreplicated' | 'under_benchmarked' | 'problem_coverage'
gaps(id, subdomain_id, kind, title, evidence TEXT, severity,
     linked_paper_ids JSONB, created_by_user_id, source, created_at)
-- source: 'manual' in v1; 'auto:<detector_name>' in Phase 2
gap_status(gap_id, user_id, status, PK(gap_id,user_id))  -- 'open'|'explored'|'dismissed'

-- Canonical entities (M3)
benchmarks(id, name, subdomain_id, frequency INT, latest_result TEXT, latest_date, paper_ids JSONB)
baselines(id, name, subdomain_id, frequency INT, paper_ids JSONB)

-- Per-user state
user_paper_state(user_id, paper_id, status, notes TEXT, updated_at, PK(user_id,paper_id))
user_follows(user_id, subdomain_id, PK(user_id,subdomain_id))
user_pins(user_id, subdomain_id, position, PK(user_id,subdomain_id))
user_prefs(user_id PK, density, theme, notification_prefs JSONB)

-- NotebookLM artifacts
notebooklm_notebooks(id, nlm_notebook_id, title, query TEXT, source_paper_ids JSONB,
                     audio_url, audio_r2_key, video_url, mindmap_url, chat_available,
                     owner_user_id, visibility, created_at)

-- Novelty
novelty_checks(id, user_id, query TEXT, subdomain_id, verdict JSONB,
               similar_paper_ids JSONB, created_at, status TEXT)  -- 'pending'|'done'|'failed'

-- Team feed — typed dispatch (not EAV). kind is enum; payload validated by discriminated pydantic union.
team_events(id, actor_user_id, kind TEXT,
            paper_id NULL, subdomain_id NULL, notebook_id NULL, gap_id NULL,
            payload JSONB, created_at)
-- kind ∈ 'notebook_created'|'audio_ready'|'novelty_run'|'gap_explored'|'paper_flagged'|'subdomain_created'

-- Signals
subdomain_signals(subdomain_id, kind, body, trend REAL, window_start, window_end,
                  PK(subdomain_id,kind,window_end))

-- Job tracking (mirror arq state in Postgres for reliable UI polling)
jobs(id PK, user_id, kind, status, payload JSONB, result JSONB, error TEXT, created_at, updated_at)
```

**Embedding column:** migrate `paper_analyses.embedding BLOB` → `vector(768)` with HNSW index. **Decision committed:** Gemini `text-embedding-004` at 768 dims. Upgrade path: add `embedding_v2 vector(N)`, dual-read during cutover, drop old column.

---

## Gaps tab — MANUAL workspace in v1

The UI's three gap kinds (`unreplicated`, `under_benchmarked`, `problem_coverage`) are correctly typed, but detecting them automatically requires data we don't yet trust:

- **unreplicated** depends on citation completeness and publication-age adjustment. Papers <6 months old will always look "uncited."
- **under_benchmarked** depends on `experiments.benchmarks` being normalized. Pre-M3, it's free text.
- **problem_coverage** depends on `problem_short` being canonicalized — it isn't.

**v1 plan:** Gaps tab is a manual curation surface. Users (especially the researcher owner) add gaps with the same three-kind typology. The UI's heatmap becomes a *reading aid* for gap authors, not a data source.

**Phase 2 (post-M5):** detectors fed by normalized benchmarks + citation-age adjustment. Ship one detector at a time, gated on a "precision@10 by human review ≥ 0.6" bar.

---

## User-created subdomain flow (explicit)

When a user creates a private topic `{name, tagline, keywords}`:

1. Embed `name + tagline + keywords` → `subdomains.embedding`.
2. **Default:** no LLM assignment. Rank corpus by cosine similarity, present top-50 to the user, they check papers in → rows in `paper_subdomains(source='manual:<user>', is_primary=false)`.
3. **Optional "Suggest more" button:** runs LLM assigner on top-50 embedding neighbors only (not all 285) — cost-capped at ~$0.10 per invocation.
4. Private subdomains don't affect any paper's `is_primary` row. Tab view unions: papers where `paper_subdomains(subdomain_id=X, is_primary=true)` OR `paper_subdomains(subdomain_id=X, source LIKE 'manual:%')`.

This avoids the "$0.90 per user topic" rerun and makes the flow feel like curation.

---

## Endpoints grouped by UI tab

### Shell / auth
- `POST /auth/google/callback` · `POST /auth/logout` · `GET /me` · `PATCH /me/prefs`
- `GET /corpus/stats`

### Today ([src/ui/tabs/today.jsx](../src/ui/tabs/today.jsx))
- `GET /today/briefing?since=…` · `GET /today/pinned-topics` · `GET /team/feed?limit=20` · `GET /audio/fresh`

### Subdomains ([subdomains.jsx](../src/ui/tabs/subdomains.jsx) / [subdomain_page.jsx](../src/ui/tabs/subdomain_page.jsx))
- `GET /subdomains` · `GET /subdomains/{id}` (bundle: papers, sota, benchmarks, baselines, labs, gaps, signals)
- `POST /subdomains` (create private topic) · `PATCH /subdomains/{id}/follow` · `/pin`
- `POST /subdomains/{id}/suggest-papers` → job_id
- `GET /subdomains/review-queue` · `POST /subdomains/review/{paper_id}`
- `POST /papers/{id}/reassign`

### Feed ([feed.jsx](../src/ui/tabs/feed.jsx))
- `GET /papers?q=&category=&significance=&page=` (lexical + semantic search)

### Novelty ([novelty.jsx](../src/ui/tabs/novelty.jsx))
- `POST /novelty` → job_id · `GET /jobs/{id}`

### Gaps ([gaps.jsx](../src/ui/tabs/gaps.jsx))
- `GET /subdomains/{id}/coverage-heatmap` — **informational only** (benchmark × method counts); UI does not auto-flag zero cells as gaps.
- `GET /gaps?subdomain=&kind=` — three kinds, manually curated in v1
- `POST /gaps` (create gap manually) · `POST /gaps/{id}/status`
- `POST /gaps/{id}/novelty-check` → job_id

### Notebook ([notebook.jsx](../src/ui/tabs/notebook.jsx))
- `POST /notebooks/candidates` — embedding similarity → ranked with reason labels
- `POST /notebooks` — persist selection + NotebookLM notebook creation
- `POST /notebooks/{id}/artifacts?kind=audio|video|mindmap|briefing` → job_id
- `GET /notebooks/{id}` · `GET /notebooks?mine=1&team=1`
- `POST /notebooks/{id}/chat` — proxy into NotebookLM chat (stretch within M2)

### Queue ([queue.jsx](../src/ui/tabs/queue.jsx))
- `GET /me/queue` · `PATCH /me/papers/{id}` (status + notes)
- `GET /team/artifacts`

### Paper drawer ([paper_drawer.jsx](../src/ui/paper_drawer.jsx))
- `GET /papers/{id}` · `POST /papers/{id}/queue` · `POST /papers/{id}/add-to-notebook/{notebook_id}`

---

## Feature pipelines (new code)

### Embeddings — [src/layer1/embedder.py](../src/layer1/embedder.py)
**Model committed:** Gemini `text-embedding-004`, 768 dims. Input: `title + brief + problem_short`. Stored as `vector(768)` with HNSW index. Back-fill 285 papers once (~$0.01 total). Hook into Layer 1 so new papers embed at ingestion.

### Subdomain assigner — [src/layer4/subdomain_assigner.py](../src/layer4/subdomain_assigner.py)
- Seed: [research_config/subdomains.yaml](../research_config/subdomains.yaml) with ~12–15 curated subdomains. **This is a researcher deliverable** that blocks M3 until the user produces it.
- LLM agent (Gemini Flash, prompt `src/layer1/prompts/subdomain_assign.txt`): given paper brief+tags and the curated catalog, return top-3 `{subdomain_id, confidence}`.
- top-1 ≥ 0.7 → write `paper_subdomains(is_primary=true, source='auto')`; else → `paper_subdomain_review` with top-3.
- Back-fill: one-time script on 285 papers.

### Benchmark normalizer — [src/layer4/benchmark_normalizer.py](../src/layer4/benchmark_normalizer.py)
LLM + fuzzy match collapses free-text benchmarks into canonical `benchmarks` rows; computes `frequency`, `latest_result`, `latest_date`, `paper_ids`. Runs weekly; feeds `sotaFor()` and `benchmarksFor()`.

### Signal detector — [src/layer4/signal_detector.py](../src/layer4/signal_detector.py)
Nightly: per subdomain, last-7d count, 4-week EMA, top new affiliations, spike flag = 7d > 2× EMA. Writes `subdomain_signals`.

### Novelty service — [src/api/services/novelty.py](../src/api/services/novelty.py)
Pipeline: (1) embed query, (2) pgvector cosine top-k within selected subdomain, (3) LLM verdict agent (prompt `src/api/services/prompts/novelty.txt`) returning:
```json
{
  "score_0_1": 0.62,
  "verdict_label": "combinatorial_novelty",
  "breakdown_ratios": {"novel": 0.3, "one_step": 0.5, "not_novel": 0.2},
  "novel_aspects": [...],
  "established": [...],
  "closest_prior": [{arxiv_id, sim}...],
  "required_baselines": [...],
  "required_benchmarks": [...]
}
```
Server **normalizes** `breakdown_ratios` to sum=1. **M4 is gated** on a held-out evaluation: 20 queries with panel-of-humans ground-truth verdict; agreement κ > 0.4 required before shipping to users.

### NotebookLM service — [src/api/services/notebooks.py](../src/api/services/notebooks.py)
Async wrapper over notebooklm-py. Operations: `create_from_papers`, `generate_artifact`, `chat`. Circuit breaker + R2 cache for audio. Long operations are arq jobs.

---

## Migration: SQLite → Postgres (M1a — budget 1 full week)

**Measured scope (2026-04-23): 77 occurrences of dialect-specific tokens across 13 files.** Plus `CREATE TABLE IF NOT EXISTS` and `PRAGMA` sites.

**Files requiring per-file review (ranked by count):**
- [src/db/database.py](../src/db/database.py) — 7 occurrences (connection helper, highest leverage)
- [src/layer3/data_collector.py](../src/layer3/data_collector.py) — 16
- [src/scripts/layer2_analyze_fronts.py](../src/scripts/layer2_analyze_fronts.py) — 12
- [src/layer0/fetch_and_score.py](../src/layer0/fetch_and_score.py) — 13
- [src/scripts/paper_lookup.py](../src/scripts/paper_lookup.py) — 14
- [src/scripts/recategorize_papers.py](../src/scripts/recategorize_papers.py) — 4
- [src/scripts/export_dashboard_data.py](../src/scripts/export_dashboard_data.py) — 3
- [src/layer1/pipeline.py](../src/layer1/pipeline.py) — 3
- [src/scripts/layer1_retag.py](../src/scripts/layer1_retag.py) — 1
- [src/layer2/front_summarizer.py](../src/layer2/front_summarizer.py) — 1
- [src/layer2/front_detection.py](../src/layer2/front_detection.py) — 1
- [src/layer2/citation_graph.py](../src/layer2/citation_graph.py) — 1
- [src/llm_client.py](../src/llm_client.py) — 1

**Per-site rewrites:**
- `INSERT OR REPLACE` → `INSERT … ON CONFLICT (key) DO UPDATE SET …`
- `json(column)` / JSON-as-TEXT → `jsonb`
- Autoincrement → `BIGSERIAL`
- Boolean stored as INT 0/1 → native `BOOLEAN`

**Embedding schema change** is its own Alembic step: `BLOB` → `vector(768)` + HNSW index + back-fill script.

**Data copy:** custom Python script `src/scripts/migrate_sqlite_to_pg.py` (not pgloader — we are promoting types).

**Concurrency:** introduce SQLAlchemy engine with pool; wrap Layer 1 per-paper analysis in a transaction.

**Local dev parity:** `docker-compose.yml` with Postgres + pgvector + Redis. No SQLite fallback in code. pytest uses testcontainers-python.

**Rollback:** every Alembic migration has working `downgrade()`; nightly `pg_dump` to R2 with 14-day retention; feature flags (env vars) on new endpoints so we can disable broken features without redeploy.

---

## Jobs, scheduling, failure handling

- **Scheduler:** arq cron tasks wrapping `pg_try_advisory_lock(hash(task_name))`. Safe under >1 instance. **Fallback:** host cron if arq scheduling proves unreliable.
- **Retries:** arq's `JobCtx` with 3 retries + exp backoff; on exhaust → `jobs.status='failed'` with error message.
- **UI contract:** every expensive endpoint returns `{job_id}` immediately. UI polls `GET /jobs/{id}` → `{status, progress?, result?, error?}`. "Retry" button reposts.
- **Rate limits** per user (Redis leaky bucket): novelty 10/h, notebook-generate 5/h, audio-generate 3/h, suggest-papers 5/h, search 20/min. 429 + `Retry-After`.
- **Observability:** Sentry; structured JSON logs; `/health` endpoint checks DB + Redis + NotebookLM circuit state + session-file mtime.

---

## NotebookLM operational runbook

**Auth break (expected every ~30–90 days):**
1. Circuit breaker opens; Sentry alert fires; UI shows "Notebook features temporarily unavailable."
2. SSH to the VPS.
3. Run `notebooklm login` in the container (headless alternative: paste cookies from browser session into session file).
4. Verify with `notebooklm list-notebooks`.
5. Circuit breaker auto-closes on next successful health check (nightly) or after manual `POST /admin/reset-breaker/notebooklm`.

**Rate-limit hit:**
- User-facing: 429 with "NotebookLM quota reached, try again in 1 hour."
- Internal: per-user rate limits (above) keep us well under Google's; if we hit anyway, it's a Google-side change and the circuit breaker trips.

---

## Milestones (realistic)

**M1a (week 1) — Postgres migration**
SQL sweep across 13 files (scoped above), data copy script, docker-compose, CI green on Postgres. **Acceptance test:** all existing Layer 1 tests pass against Postgres; pg_dump → restore to fresh DB reproduces identical row counts.

**M1b (weeks 2–5) — Foundation API**
FastAPI shell + OAuth + invites + JWT cookies + user state tables + read-only endpoints + lexical corpus search + frontend rewrite replacing [data.jsx](../src/ui/data.jsx) + **parameterization of hardcoded subdomain IDs** in [app.jsx](../src/ui/app.jsx), [today.jsx](../src/ui/tabs/today.jsx), [queue.jsx](../src/ui/tabs/queue.jsx), [novelty.jsx](../src/ui/tabs/novelty.jsx), [gaps.jsx](../src/ui/tabs/gaps.jsx). Deploy to VPS. **Acceptance tests:**
- Non-invited email → 403 at callback.
- User state persists across sessions (Playwright smoke).
- UI loads with no references to `window.PAPERS`; network tab shows `/api/*` calls.

**M1c (week 6) — Embeddings**
Gemini `text-embedding-004` back-fill; pgvector HNSW; hybrid lexical+embedding search. **Acceptance test:** semantic search for "evolutionary neural architecture" returns AlphaEvolve paper; latency p95 <300ms.

**M2 (weeks 7–10) — NotebookLM integration**
`notebooklm_client` async wrapper + circuit breaker + health check + R2 cache; notebook CRUD; `/notebooks/candidates`; notebook creation on save; audio/video/mindmap arq jobs; Notebook tab and paper drawer "spawn notebook" live. Chat proxy as stretch.
**Acceptance tests:**
- `POST /notebooks` with 5 paper IDs creates a real NotebookLM notebook (row has valid `nlm_notebook_id`).
- `POST /notebooks/{id}/artifacts?kind=audio` completes within 5 min; audio plays from R2 cache even after Google URL rotates.
- Kill NotebookLM network access → breaker trips within 3 calls → UI shows unavailable.

**M3 (weeks 11–13) — Taxonomy & structured data**
Researcher delivers `subdomains.yaml`; subdomain assigner + back-fill; review queue UI; benchmark normalizer; signal detector. Subdomains / SubdomainPage / Gaps tabs real (Gaps tab = manual workspace). Hardcoded subdomain IDs fully removed from frontend logic (verified by a CI grep check).
**Acceptance tests:**
- All 285 papers have either `is_primary=true` subdomain row or a `paper_subdomain_review` entry.
- Low-confidence banner count is live.
- CI guard fails if any `.jsx` under `src/ui/` contains `evo_llm_search|nl_to_opt|llm_serving_opt|rl_for_opt_modeling` outside `source_data/`.

**M4 (weeks 14–16) — On-demand LLM features**
Novelty endpoint + verdict-quality evaluation gate (κ > 0.4); user-created private topics with embedding-match flow; "Suggest more papers" LLM top-up.
**Acceptance tests:**
- Verdict eval on 20-query held-out set passes κ > 0.4 vs. panel.
- `breakdown_ratios` server-normalized to sum=1 ± 0.001.
- Private topic creation → suggest-more → user adds 12 → appear in subdomain view.

**M5 (weeks 17–18) — Social & audio loop**
Team feed writes; shared artifacts browse; daily/weekly NotebookLM Audio Overview per followed subdomain; digest email reusing Layer 3 renderer.
**Acceptance tests:**
- Gap explored by user A visible in Team Feed for user B within 5s.
- Play button on Today tab plays live NotebookLM Audio Overview from R2 cache.

**Total realistic timeline: ~4–5 months** for one developer, single-threaded. Aggressive milestones can overlap if NotebookLM auth doesn't stall.

---

## Cost envelope (team of 5, steady state)

See [infra-plan.md](infra-plan.md#monthly-cost-summary) for the full infrastructure cost breakdown. Per-feature LLM costs:

| Item | Volume / month | Unit | Cost |
|---|---|---|---|
| Layer 1 analysis | 300 papers | $0.011 | $3.3 |
| Embeddings | 300 papers | negligible | ~$0 |
| Subdomain assign | 300 papers | $0.003 | $0.9 |
| Novelty checks | 80 | $0.02 | $1.6 |
| NotebookLM artifacts | 40 | free tier | $0* |

*NotebookLM rate limits apply — 40 audio-generates/month is well within Google's free tier for a team account, but heavy months could hit daily caps. If the team routinely hits them, the fallback is to introduce a paid TTS path (OpenAI TTS) as an opt-in fallback — out of scope for v1.

---

## Critical files to create / modify

**New**
- [src/api/main.py](../src/api/main.py), [settings.py](../src/api/settings.py), [deps.py](../src/api/deps.py)
- [src/api/auth/](../src/api/auth/) (google.py, jwt.py, invites.py, rate_limit.py)
- [src/api/routers/](../src/api/routers/) (papers, subdomains, gaps, notebooks, novelty, queue, me, team, jobs)
- [src/api/services/](../src/api/services/) (novelty, notebooks, notebooklm_client, search, subdomain_assign)
- [src/api/schemas/](../src/api/schemas/) (pydantic response models)
- [src/layer1/embedder.py](../src/layer1/embedder.py) · [src/layer1/prompts/subdomain_assign.txt](../src/layer1/prompts/subdomain_assign.txt) · [src/api/services/prompts/novelty.txt](../src/api/services/prompts/novelty.txt)
- [src/layer4/](../src/layer4/) (subdomain_assigner, benchmark_normalizer, signal_detector)
- [src/jobs/](../src/jobs/) (worker [arq], tasks, scheduler)
- [src/db/migrations/](../src/db/migrations/) (Alembic)
- [src/scripts/migrate_sqlite_to_pg.py](../src/scripts/migrate_sqlite_to_pg.py)
- [research_config/subdomains.yaml](../research_config/subdomains.yaml) — researcher deliverable
- `Dockerfile`, `docker-compose.yml`, `Caddyfile` (see [infra-plan.md](infra-plan.md))

**Modify (SQL dialect sweep — 13 files, 77 sites)**
- [src/db/database.py](../src/db/database.py), [src/db/schema.sql](../src/db/schema.sql)
- [src/layer0/fetch_and_score.py](../src/layer0/fetch_and_score.py)
- [src/layer1/pipeline.py](../src/layer1/pipeline.py)
- [src/layer2/citation_graph.py](../src/layer2/citation_graph.py), [front_detection.py](../src/layer2/front_detection.py), [front_summarizer.py](../src/layer2/front_summarizer.py)
- [src/layer3/data_collector.py](../src/layer3/data_collector.py)
- [src/llm_client.py](../src/llm_client.py)
- [src/scripts/paper_lookup.py](../src/scripts/paper_lookup.py), [export_dashboard_data.py](../src/scripts/export_dashboard_data.py), [layer1_retag.py](../src/scripts/layer1_retag.py), [layer2_analyze_fronts.py](../src/scripts/layer2_analyze_fronts.py), [recategorize_papers.py](../src/scripts/recategorize_papers.py)

**Modify (frontend — parameterize hardcoded subdomain IDs)**
- [src/ui/app.jsx:8](../src/ui/app.jsx) — default `sdId` from server
- [src/ui/tabs/today.jsx:9,111](../src/ui/tabs/today.jsx) — pinned IDs from `/today/pinned-topics`
- [src/ui/tabs/queue.jsx:81](../src/ui/tabs/queue.jsx) — hardcoded list from `/me/follows` or user prefs
- [src/ui/tabs/novelty.jsx:5](../src/ui/tabs/novelty.jsx) — default from `/subdomains`
- [src/ui/tabs/gaps.jsx:9,10](../src/ui/tabs/gaps.jsx) — current selection from URL param or user prefs
- [src/ui/data.jsx](../src/ui/data.jsx) — delete, replace with `src/ui/api_client.js`
- [src/ui/paper_drawer.jsx](../src/ui/paper_drawer.jsx) — wire queue/notes/add-to-notebook/generate-audio to API

**requirements.txt additions**
- `fastapi`, `uvicorn[standard]`, `pydantic-settings`, `sqlalchemy[asyncio]`, `asyncpg`, `alembic`, `pgvector`, `authlib`, `arq`, `pybreaker`, `sentry-sdk`, `redis`, `httpx`, `testcontainers`, `notebooklm-py`, `boto3` (R2)

**Out of scope (required to ship eventually but not this plan)**
- Optimistic UI for queue drag/drop; accessibility pass; mobile layout.
- CI/CD workflow files.
- Team-growth schema (teams table, org-level visibility, `shared` subdomains) — add when team > 10.
- Automated gap detectors (Phase 2).
- Paid TTS fallback.

---

## Verification (global — in addition to per-milestone gates)

**End-to-end sanity:** wear each tab for 3 minutes, confirm no button is a no-op, no data is hardcoded, network tab shows every render fetches from `/api/*`, and `grep -r 'window.PAPERS\|window.SUBDOMAINS\|window.GAPS\|window.SIGNALS' src/ui/` returns empty.
