# Research Intelligence System - Architecture Guide

This system transforms ArXiv paper aggregation into deep research intelligence through a four-layer architecture. Each layer builds on the previous one, creating progressively richer understanding.

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 0: FETCH & SCORE                                          │
│ ├─ ArXiv keyword search + Semantic Scholar citation crawling   │
│ ├─ LLM relevance scoring (0-10 scale)                          │
│ ├─ Code link discovery (Papers with Code, GitHub)              │
│ └─ JSON/Markdown output (README.md, docs/*.json)               │
└─────────────────────────────────────────────────────────────────┘
            ↓ (arxiv_id, title, abstract, category)
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: DEEP ANALYSIS (4 agents)                              │
│ ├─ PDF extraction (pypdf + caching)                            │
│ ├─ Reader: problem, methodology, experiments, results          │
│ ├─ Methods Extractor: tags, lineage, frameworks                │
│ ├─ Positioning: relevance to researcher's active projects      │
│ └─ Database storage (paper_analyses table)                     │
└─────────────────────────────────────────────────────────────────┘
            ↓ (methods, problems, positioning, artifacts)
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: BIBLIOMETRIC FRONTS                                   │
│ ├─ Citation graph (Semantic Scholar API, per category)         │
│ ├─ Co-citation network (Salton's cosine, Louvain clustering)   │
│ ├─ Front summarization (LLM generates trend descriptions)      │
│ ├─ Bridge paper detection (papers connecting fronts)           │
│ └─ Database storage (research_fronts, cocitation_edges)        │
└─────────────────────────────────────────────────────────────────┘
            ↓ (fronts, trends, bridges)
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: LIVING REVIEWS & UPDATES                              │
│ ├─ Daily updates (append new papers to living review)          │
│ ├─ Weekly revisions (restructure, add narrative)               │
│ ├─ Monthly rewrites (major restructuring)                      │
│ ├─ Email generation (daily briefings, weekly deep dives)       │
│ └─ Database storage (review_updates audit trail)               │
└─────────────────────────────────────────────────────────────────┘
            ↓
     📧 Email + 📄 Living Reviews (docs/living_reviews/)
```

## Layer Responsibilities

### Layer 0: Fetch & Score ([layer0/](layer0/))

**Purpose:** Acquire relevant papers and filter by relevance

**Input:** ArXiv categories, keywords, seed papers (from [../research_config/](../research_config/))
**Output:** JSON files with paper metadata, [README.md](../README.md)

**Files:**
- [layer0/fetch_and_score.py](layer0/fetch_and_score.py) - Main orchestrator (renamed from `daily_arxiv.py`)

**Key Features:**
- **Dual search strategy**:
  - Keyword-based: ArXiv API with OR-combined filters
  - Citation-based: Semantic Scholar citation graph crawling from seed papers
- **LLM relevance scoring**: Each paper scored 0-10 using category description + researcher context
- **Score caching**: `docs/relevance_cache.json` + database to avoid redundant API calls
- **Code link discovery**: Papers with Code API → GitHub search fallback
- **Resilient API calls**: Retries, timeouts, rate limiting (ArXiv: 3 retries, 3s backoff; S2: 5s on 429)

**When to run:**
```bash
# Fetch all categories
python src/layer0/fetch_and_score.py

# Fetch single category
python src/layer0/fetch_and_score.py --category "LLMs for Algorithm Design"

# Update code links only (no new papers, no LLM scoring)
python src/layer0/fetch_and_score.py --update_paper_links
```

**Configuration:** See [../research_config/README.md](../research_config/README.md)

---

### Layer 1: Deep Analysis ([layer1/](layer1/))

**Purpose:** Extract detailed understanding from paper PDFs using multi-agent analysis

**Input:** ArXiv IDs from Layer 0
**Output:** Structured analysis in database (`paper_analyses` table)

**Architecture:** 4-agent pipeline with LLM structured output

1. **Reader Agent** - Extracts:
   - Problem addressed
   - Methodology (algorithms, techniques)
   - Experiments & results
   - Artifacts (code, datasets, benchmarks)

2. **Methods Extractor** - Identifies:
   - Method/problem tags (for categorization)
   - Lineage: "builds on X", "extends Y", "combines A and B"
   - Frameworks/libraries used

3. **Positioning Agent** - Assesses:
   - Relevance to researcher's active projects (0-10 scale)
   - Explanation of why it matters (or doesn't)
   - Connections to researcher's methodologies

4. **Synthesis Controller** - Merges outputs and stores in database

**Files:**
- [layer1/pipeline.py](layer1/pipeline.py) - 4-agent orchestrator
- [layer1/pdf_extractor.py](layer1/pdf_extractor.py) - PDF download and text extraction
- [layer1/schemas.py](layer1/schemas.py) - Pydantic schemas for structured output
- [layer1/prompts/](layer1/prompts/) - Agent prompt templates

**Key Features:**
- **PDF caching** (`cache/pdfs/`) - Download once, reuse forever
- **Gemini structured output** - Reliable JSON parsing via Pydantic schemas
- **Per-category researcher context** - Personalized relevance via `research_config/researcher_profile.md`
- **Database persistence** - Downstream layers query this data

**When to run:**
```bash
# Analyze new papers in category
python src/scripts/layer1_analyze_new.py --category "LLMs for Algorithm Design" --max 20

# Re-analyze specific paper
python src/scripts/layer1_analyze_new.py --arxiv-id 2401.12345

# Analyze all unanalyzed papers
python src/scripts/layer1_analyze_new.py --all
```

**Cost:** ~$0.01 per paper (3 agents × $0.003-0.004 each with Gemini Flash)

---

### Layer 2: Bibliometric Fronts ([layer2/](layer2/))

**Purpose:** Detect research fronts through citation network analysis

**Input:** Paper analyses from Layer 1
**Output:** Research fronts, trend summaries, bridge papers (database)

**Algorithm:**
1. **Build directed citation graph** - Semantic Scholar API (citing & cited-by)
2. **Construct co-citation network** - Salton's cosine similarity between papers
3. **Louvain community detection** - Identify research fronts (clusters)
4. **LLM summarization** - Generate trend description for each front
5. **Bridge detection** - Find papers connecting multiple fronts

**Files:**
- [layer2/citation_graph.py](layer2/citation_graph.py) - Directed citation graph builder
- [layer2/front_detection.py](layer2/front_detection.py) - Co-citation network + Louvain
- [layer2/front_summarizer.py](layer2/front_summarizer.py) - LLM trend summarization
- [layer2/bridge_papers.py](layer2/bridge_papers.py) - Cross-front connection detection

**Key Features:**
- **Per-category graphs** - Separate front detection per research topic
- **Configurable clustering** - `--min-front-size`, `--min-similarity` for tuning
- **Semantic Scholar API** - With retries and rate limiting
- **Front lineage tracking** - Evolution of research trends over time

**When to run:**
```bash
# Detect fronts for category
python src/scripts/layer2_detect_fronts.py \
  --category "LLMs for Algorithm Design" \
  --min-front-size 2 \
  --min-similarity 0.9

# Analyze existing fronts
python src/scripts/layer2_analyze_fronts.py --category "LLMs for Algorithm Design"

# Visualize citation graph
python src/scripts/layer2_visualize.py --category "LLMs for Algorithm Design"
```

**Cost:** ~$0.002 per front for LLM summarization (using Gemini Flash)

---

### Layer 3: Living Reviews & Updates ([layer3/](layer3/))

**Purpose:** Maintain living literature reviews with daily/weekly/monthly update cycles

**Input:** Papers from Layer 1, fronts from Layer 2
**Output:** Markdown reviews (`docs/living_reviews/`), HTML emails

**Update Cycles:**
- **Daily:** Append new papers to existing review structure (minimal changes)
- **Weekly:** Restructure sections, add narrative, highlight research fronts
- **Monthly:** Major rewrite with new organization

**Files:**
- [layer3/daily_update.py](layer3/daily_update.py) - Append new papers
- [layer3/weekly_revision.py](layer3/weekly_revision.py) - Restructure + narrative
- [layer3/email_renderer.py](layer3/email_renderer.py) - Generate HTML emails
- [layer3/email_sender.py](layer3/email_sender.py) - Gmail SMTP delivery
- [layer3/graph_visualizer.py](layer3/graph_visualizer.py) - Front visualization

**Key Features:**
- **Separate living review per category** - One Markdown file per research topic
- **Unified daily email** - All categories in one briefing
- **Per-category weekly emails** - Deep dive into each research area
- **Citation graph visualizations** - Embedded in emails
- **Audit trail** - `review_updates` table tracks all changes

**When to run:**
```bash
# Daily briefing (all categories, last 7 days)
python src/scripts/layer3_run.py daily --days 7

# Weekly deep dive (single category, last 7 days)
python src/scripts/layer3_run.py weekly \
  --category "LLMs for Algorithm Design" \
  --days 7

# Monthly rewrite
python src/scripts/layer3_run.py monthly --category "LLMs for Algorithm Design"
```

**Cost:** ~$0.001 per update for email generation (using Gemini Flash)

---

## Configuration ([../research_config/](../research_config/))

All domain-specific configuration consolidated in one place:

| File | Purpose | What to Change for New Domain |
|------|---------|------------------------------|
| [research_domain.yaml](../research_config/research_domain.yaml) | Categories, filters, outputs | ✅ **YES** - Define your categories |
| [researcher_profile.md](../research_config/researcher_profile.md) | Per-category research context | ✅ **YES** - Describe your interests |
| [constants.yaml](../research_config/constants.yaml) | MIN_YEAR, thresholds, APIs | Maybe - Usually keep defaults |
| [model_config.yaml](../research_config/model_config.yaml) | LLM assignments per agent | Maybe - Usually keep defaults |
| [README.md](../research_config/README.md) | Adaptation guide | Read first! |

**To adapt to a new research domain (e.g., "AI for Healthcare"):**
1. Edit `research_config/research_domain.yaml` - define categories and filters
2. Edit `research_config/researcher_profile.md` - describe your research focus
3. Run Layer 0 to start fetching papers
4. Optionally run Layer 1-3 for deeper intelligence

**No code changes needed!**

---

## Database Schema ([db/research_intelligence.db](db/research_intelligence.db))

| Table | Layer | Purpose |
|-------|-------|---------|
| `paper_analyses` | 1 | Deep analysis results (4 agents) |
| `citations` | 2 | Directed citation graph edges |
| `cocitation_edges` | 2 | Co-citation network |
| `research_fronts` | 2 | Detected research trends |
| `front_lineage` | 2 | Front evolution over time |
| `bridge_papers` | 2 | Papers connecting fronts |
| `review_updates` | 3 | Living review audit trail |

**Database location:** Defined in [config.py](config.py) as `DB_PATH`

---

## Typical Workflows

### Daily Research Briefing (Automated via GitHub Actions)

**Workflow:** [.github/workflows/bidaily_email.yml](../.github/workflows/bidaily_email.yml)

```yaml
Trigger: Sun/Tue/Thu at 10 AM PST
Steps:
  1. Layer 0: Fetch papers for each category
  2. Layer 1: Deep analysis of new papers (max 20 per category)
  3. Layer 3: Generate daily briefing email (unified across categories)
  4. Commit: README, JSON, cache, database
```

### Weekly Research Report (Automated via GitHub Actions)

**Workflow:** [.github/workflows/weekly_report.yml](../.github/workflows/weekly_report.yml)

```yaml
Trigger: Tuesday at 11 AM UTC
Steps:
  1. Layer 0: Fetch papers for each category
  2. Layer 1: Deep analysis of new papers
  3. Layer 2: Detect research fronts (per category)
  4. Layer 3: Generate weekly deep-dive emails (separate per category)
  5. Commit: README, fronts, living reviews, database
```

### Manual Deep Dive (Local Development)

```bash
# 1. Fetch papers
python src/layer0/fetch_and_score.py --category "LLMs for Algorithm Design"

# 2. Analyze deeply
python src/scripts/layer1_analyze_new.py --category "LLMs for Algorithm Design" --max 50

# 3. Detect fronts
python src/scripts/layer2_detect_fronts.py --category "LLMs for Algorithm Design"

# 4. Update living review
python src/scripts/layer3_run.py weekly --category "LLMs for Algorithm Design" --days 14
```

---

## Extension Points

### Adding a New Layer

Layers are independent - you can add Layer 4 (e.g., "Personalized Recommendations") without modifying existing layers:

1. Create `src/layer4/` directory
2. Read from database tables populated by Layer 1-3
3. Add CLI script to `src/scripts/layer4_*.py`
4. Update workflows if needed

### Switching LLM Providers

Edit [../research_config/model_config.yaml](../research_config/model_config.yaml):

```yaml
relevance_scorer:
  provider: "anthropic"  # Change from "openai"
  model: "claude-haiku-4-5-20251001"
  temperature: 0.0
```

### Adding New Search Strategies

Extend Layer 0 with new paper sources:

1. Add new module to `src/layer0/` (e.g., `pubmed_search.py`)
2. Call from `fetch_and_score.py` orchestrator
3. Add strategy to `research_config/research_domain.yaml`:

```yaml
categories:
  "AI for Healthcare":
    search_strategy: "pubmed"  # New strategy
    pubmed_query: "(artificial intelligence) AND (clinical trials)"
```

---

## Cost Estimates

Per 100 papers (with Gemini Flash):

| Layer | Cost | What it does |
|-------|------|--------------|
| **Layer 0** | $0.05 | Relevance scoring only (100 papers × $0.0005) |
| **Layer 1** | $1.00 | 3 agents × 100 papers × $0.003 each |
| **Layer 2** | $0.20 | Front summarization (~10 fronts × $0.02) |
| **Layer 3** | $0.10 | Email generation (daily + weekly) |
| **Total** | **$1.35** | Full pipeline for 100 papers |

**Cost optimization tips:**
- Use `--max` flag to limit Layer 1 analysis
- Use Gemini Flash instead of Pro (10x cheaper)
- Cache PDFs and scores to avoid re-processing
- Adjust `relevance_threshold` to filter more aggressively

---

## Questions?

**See also:**
- [../research_config/README.md](../research_config/README.md) - How to adapt to new domains
- [../.claude/CLAUDE.md](../.claude/CLAUDE.md) - Claude Code integration guide
- [../docs/PLAN.md](../docs/PLAN.md) - Implementation history

**Contact:** Open an issue on GitHub or check project documentation.
