# Research Intelligence System - Implementation Progress

**Last Updated:** 2026-02-14
**Project:** Automated Research Intelligence with Living Literature Reviews
**LLM Provider:** Gemini (gemini-2.5-flash with structured output)

---

## Overview

Building a three-layer system that transforms ArXiv paper aggregation into deep research intelligence:
- **Layer 1:** Multi-agent deep paper analysis (4 agents) ✅ **COMPLETED**
- **Layer 2:** Bibliometric front detection (citation graphs, co-citation networks) ✅ **COMPLETED**
- **Layer 3:** Living literature reviews + unified email (3 separate reviews, 1 email) ✅ **COMPLETED**

**Key Design:** Three separate living reviews (one per category), unified daily email with graph visualizations.

---

## Implementation Status

### ✅ Phase 1: Foundation & Database (COMPLETED)

| Task | Status | Files Created |
|------|--------|---------------|
| Database schema design | ✅ Complete | `src/db/schema.sql` |
| Database access layer | ✅ Complete | `src/db/database.py` |
| Database initialization test | ✅ Complete | 8 tables verified |
| Layer 1 folder structure | ✅ Complete | `src/layer1/`, `src/layer1/prompts/` |
| Agent prompts | ✅ Complete | `reader.txt`, `methods.txt`, `positioning.txt`, `researcher_profile.md` |
| Pydantic schemas | ✅ Complete | `src/layer1/schemas.py` |

**Database Tables Created:**
- ✅ `paper_analyses` - Layer 1 deep analysis results
- ✅ `citations` - Citation graph edges (per category)
- ✅ `cocitation_edges` - Co-citation network
- ✅ `research_fronts` - Detected research fronts
- ✅ `front_lineage` - Front evolution tracking
- ✅ `bridge_papers` - Papers connecting fronts
- ✅ `review_updates` - Living review audit trail

---

### ✅ Phase 2: Layer 1 - Deep Paper Analysis (COMPLETED ✅)

| Task | Status | Files |
|------|--------|-------|
| PDF text extractor | ✅ Complete | `src/layer1/pdf_extractor.py` |
| Unified LLM client | ✅ Complete | `src/llm_client.py` |
| Centralized config system | ✅ Complete | `src/config.py` |
| 4-agent pipeline orchestrator | ✅ Complete | `src/layer1/pipeline.py` |
| CLI script for Layer 1 | ✅ Complete | `src/scripts/layer1_analyze_new.py` |
| Update requirements.txt | ✅ Complete | `requirements.txt` (google-genai) |
| Gemini structured output | ✅ Complete | Working with `gemini-2.5-flash` |
| End-to-end testing | ✅ Complete | Successfully analyzed test paper |
| Skill documentation | ✅ Complete | `docs/gemini-structured-output-skill.md` |

**Agent Architecture:**

1. ✅ **Reader Agent** - Extracts problem, methodology, experiments, results, artifacts
2. ✅ **Methods Extractor** - Tags methods/problems, identifies lineage
3. ✅ **Positioning Agent** - Assesses relevance to researcher's work
4. ✅ **Synthesis Controller** - Merges outputs and stores in database

**Key Implementation Details:**

- **LLM Provider:** Google Gemini (`gemini-2.5-flash`)
- **Structured Output:** Native JSON schema validation via `google-genai` SDK
- **Token Limit:** 8192 tokens per response (sufficient for complex schemas)
- **Model Selection:** Centralized in `src/config.py` - easy to mix providers per agent
- **Cost:** ~$0.01 per paper (3 agents × ~$0.003-0.004 each)

**Tested and Working:**

✅ PDF extraction and caching
✅ Reader agent with structured JSON output
✅ Methods extractor with proper schema validation
✅ Positioning agent with relevance scoring
✅ Database storage with all fields
✅ Full pipeline: PDF → 3 agents → database

---

### ✅ Phase 3: Layer 2 - Bibliometric Analysis (COMPLETED ✅)

| Task | Status | Files |
|------|--------|-------|
| Citation graph builder | ✅ Complete | `src/layer2/citation_graph.py` |
| Co-citation network + Louvain | ✅ Complete | `src/layer2/front_detection.py` |
| Front summarization (LLM) | ✅ Complete | `src/layer2/front_summarizer.py` |
| Bridge paper detection | ✅ Complete | `src/layer2/bridge_papers.py` |
| Layer 2 CLI script | ✅ Complete | `src/scripts/layer2_detect_fronts.py` |
| Layer 2 analysis script | ✅ Complete | `src/scripts/layer2_analyze_fronts.py` |
| Bug fixes (6 bugs, code review) | ✅ Complete | All Layer 2 files |

**Key Features:**
- Semantic Scholar API integration with rate limiting and retries
- Separate directed citation graphs per category (NetworkX)
- Co-citation network with Salton's cosine normalization
- Louvain community detection (configurable resolution)
- Front evolution tracking with Jaccard similarity matching
- Bridge paper detection with **weighted** cross-front edge scoring
- LLM narrative summaries + names + future directions per front (Gemini 2.5 Flash)
- Per-paper atomic fetch log + edge insertion (crash-safe)
- CLI flags: `--skip-citations`, `--skip-summaries`, `--min-front-size`, `--resolution`, `--semantic-weight`

**Current Paper Similarity Signal (Layer 2):**

> Papers are connected in the co-citation network using three signals:
>
> 1. **Co-citation** — two corpus papers cited together by an external paper
> 2. **Bibliographic coupling** — two corpus papers share common references (optional, `--bib-coupling-factor`)
> 3. **Tag similarity** — IDF-weighted Jaccard over Layer 1 method/problem tags + bonus signals for class, LLM role, lineage, domain, coupling type (`--semantic-weight`)
>
> ⚠️ **TODO (Future Enhancement):** Replace or augment tag similarity with **abstract/full-text embedding similarity** (e.g., `text-embedding-3-small` or Gemini embeddings). Store embeddings in `paper_analyses.embedding` (column already exists in schema). Use cosine similarity over embeddings as the primary semantic signal instead of tag Jaccard. This will capture relationships that tag overlap misses (paraphrased concepts, domain transfers, papers with different vocabulary for the same method).
> See: **Phase 6 — Future Enhancements** below.

---

### ✅ Phase 4: Layer 3 - Living Reviews & Email (COMPLETED ✅)

| Task | Status | Files |
|------|--------|-------|
| Data collector (Layer 1+2 aggregation) | ✅ Complete | `src/layer3/data_collector.py` |
| Graph visualization | ✅ Complete | `src/layer3/graph_visualizer.py` |
| HTML email renderer (daily/weekly/monthly) | ✅ Complete | `src/layer3/email_renderer.py` |
| Email sender (SendGrid + file fallback) | ✅ Complete | `src/layer3/email_sender.py` |
| Daily review updater | ✅ Complete | `src/layer3/daily_update.py` |
| Weekly review revision (front-based) | ✅ Complete | `src/layer3/weekly_revision.py` |
| Unified CLI | ✅ Complete | `src/scripts/layer3_run.py` |

**Output Files:**
- `docs/living_reviews/llm_for_algo.md` (Category 1)
- `docs/living_reviews/genai_for_or.md` (Category 2)
- `docs/living_reviews/or_for_genai.md` (Category 3)
- `docs/living_reviews/graphs/*.png` (Network visualizations)
- `docs/living_reviews/emails/*.html` (Email archive)

**Weekly Email — Research Landscape Section (Layer 2 inputs):**

- Front overview table: name, size, status badge, density
- Per-front detail cards: LLM summary, paper list with method/LLM-role
- Unique vs shared methods per front
- Method overlap matrix (color-coded HTML table)
- Bridge paper highlights with synthesis verdict
- Embedded front network graph (base64 PNG inline)

---

### ⏳ Phase 5: Automation & Deployment (PENDING)

| Task | Status | Files |
|------|--------|-------|
| GitHub Actions daily pipeline | ⏳ Pending | `.github/workflows/research_intelligence.yml` |
| GitHub Actions weekly fronts | ⏳ Pending | `.github/workflows/weekly_fronts.yml` |
| End-to-end local testing | ⏳ Pending | Full pipeline test |
| Configure GitHub secrets | ⏳ Pending | `ANTHROPIC_API_KEY`, `SENDGRID_API_KEY`, `EMAIL_TO` |
| Deploy and monitor | ⏳ Pending | Production deployment |

---

## Quick Reference

### Project Structure

```
llm-arxiv-daily-main/
├── src/                           # NEW: All implementation code
│   ├── db/                        # ✅ Database layer
│   │   ├── schema.sql
│   │   ├── database.py
│   │   └── research_intelligence.db
│   ├── layer1/                    # 🔄 Deep paper analysis (MAS)
│   │   ├── pdf_extractor.py       # ✅
│   │   ├── pipeline.py            # ⏳
│   │   ├── schemas.py             # ✅
│   │   └── prompts/               # ✅ All 4 agent prompts
│   ├── layer2/                    # ⏳ Bibliometric analysis
│   ├── layer3/                    # ⏳ Living reviews + email
│   └── scripts/                   # ⏳ CLI tools
├── daily_arxiv.py                 # Existing pipeline (unchanged)
├── config.yaml                    # Existing config
├── docs/                          # Existing outputs
├── requirements.txt               # Needs updating
└── PLAN.md                        # This file
```

### Three Categories (Separate Analysis)

1. **LLMs for Algorithm Design** (51 papers)
   - Keyword-based ArXiv search
   - Focus: LLMs for automatic algorithm design, evolutionary search

2. **Generative AI for OR** (143 papers)
   - Citation-based discovery (Semantic Scholar)
   - Focus: Using GenAI for optimization modeling

3. **OR for Generative AI** (108 papers)
   - Complex boolean search
   - Focus: OR formulations for LLM systems (GPU scheduling, inference optimization)

### Cost Estimates (Using Gemini 2.5 Flash)

| Component | Details | Monthly Cost |
|-----------|---------|--------------|
| **Layer 1** | 3 agents/paper × 10 papers/day × 30 days | ~$9 |
| ├─ Reader | ~15K input, 3K output tokens | $0.011/paper |
| ├─ Methods | ~20K input, 2K output tokens | $0.011/paper |
| └─ Positioning | ~25K input, 1.5K output tokens | $0.012/paper |
| **Layer 2** | Front summaries (15 fronts × 4 weeks) | ~$2 |
| **Layer 3** | Daily/weekly/monthly updates | ~$4 |
| **Total** | | **~$15/month** |

**Cost Savings:** Gemini is 3× cheaper than Anthropic Claude (~$45 → ~$15/month)

**Per-Paper Cost Breakdown:**
- Input: ~20K tokens avg × $0.30/1M = $0.006
- Output: ~2K tokens avg × $2.50/1M = $0.005
- **Total: ~$0.011 per paper** (vs ~$0.05 with Claude)

---

## Next Immediate Steps

### ✅ Completed

1. ✅ Move folders to `src/`
2. ✅ Create `PLAN.md`
3. ✅ Implement `src/layer1/pipeline.py` (4-agent orchestrator)
4. ✅ Update `requirements.txt` with new dependencies
5. ✅ Create `src/scripts/layer1_analyze_new.py` CLI
6. ✅ Test Layer 1 on real papers from existing JSON
7. ✅ Migrate from Anthropic to Gemini with structured output
8. ✅ Create Gemini structured output skill document

### ✅ Completed: Layer 2 - Bibliometric Analysis

1. ✅ **Citation graph builder** (`src/layer2/citation_graph.py`)
2. ✅ **Co-citation network + front detection** (`src/layer2/front_detection.py`)
3. ✅ **Bridge paper detection** (`src/layer2/bridge_papers.py`)
4. ✅ **Front summarization** (`src/layer2/front_summarizer.py`)
5. ✅ **Layer 2 CLI** (`src/scripts/layer2_detect_fronts.py`)

### ✅ Completed: Layer 3 - Living Reviews & Email

1. ✅ **Data collector** (`src/layer3/data_collector.py`)
2. ✅ **Graph visualizer** (`src/layer3/graph_visualizer.py`)
3. ✅ **HTML email renderer** (`src/layer3/email_renderer.py`) — daily / weekly (with front analysis) / monthly
4. ✅ **Email sender** (`src/layer3/email_sender.py`) — SendGrid + local file fallback
5. ✅ **Daily review updater** (`src/layer3/daily_update.py`)
6. ✅ **Weekly review revision** (`src/layer3/weekly_revision.py`)
7. ✅ **Unified CLI** (`src/scripts/layer3_run.py`)

### 🎯 Next: Phase 5 — Automation & Deployment

1. **GitHub Actions daily pipeline** — run Layer 1 + Layer 3 daily email
2. **GitHub Actions weekly pipeline** — run Layer 2 + weekly email with fronts
3. **End-to-end local test** — `python src/scripts/layer3_run.py weekly --preview`
4. **Configure secrets** — `GEMINI_API_KEY`, `SENDGRID_API_KEY`, `EMAIL_TO`, `EMAIL_FROM`
5. **Deploy and monitor**

---

## Testing Checklist

### Layer 1 Testing ✅ COMPLETE
- [x] Database initialization works
- [x] PDF extractor downloads and caches correctly
- [x] Reader agent produces valid JSON
- [x] Methods agent produces valid JSON
- [x] Positioning agent produces valid JSON
- [x] Full pipeline stores analysis in database
- [x] CLI script processes all unanalyzed papers
- [x] Gemini structured output prevents schema mismatches
- [x] Token limits properly configured (8192 tokens)

### Layer 2 Testing
- [ ] Semantic Scholar API integration works
- [ ] Citation graph builds correctly per category
- [ ] Co-citation network construction works
- [ ] Louvain clustering detects meaningful fronts
- [ ] Bridge paper detection identifies connectors
- [ ] Front summaries are coherent

### Layer 3 Testing
- [ ] Graph visualization generates readable PNGs
- [ ] Daily update performs surgical edits
- [ ] Weekly revision restructures correctly
- [ ] Monthly rewrite produces quality content
- [ ] Unified email has 3 sections + graphs
- [ ] Email sends successfully via SendGrid

---

## Critical Notes

- **Researcher Profile**: Must customize `src/layer1/prompts/researcher_profile.md` with actual research interests
- **Category Separation**: All three categories maintain independent graphs, fronts, and reviews
- **Unified Email**: Single daily email with three sections (one per category)
- **Graph Embeddings**: Each category's front network embedded as PNG in email
- **Integration**: System integrates with existing `daily_arxiv.py` without modifying it

---

## Timeline

- **Week 1:** ✅ Foundation & Database (DONE)
- **Week 2-3:** ✅ Layer 1 implementation (COMPLETE)
- **Week 4:** ✅ Layer 2 bibliometric analysis (COMPLETE)
- **Week 5:** ✅ Layer 3 living reviews + email (COMPLETE)
- **Week 6:** 🎯 Automation & deployment (NEXT)
- **Future:** ⏳ Embedding-based similarity (Phase 6)

**Current Progress:** ~90% complete (Phases 1–4 COMPLETE, deployment remaining)

---

---

## Phase 6: Future Enhancements (Post-Deployment)

### 🔮 Embedding-Based Paper Similarity (Layer 2 Upgrade)

**Current state:** Paper similarity in the co-citation network uses IDF-weighted tag Jaccard + structural bonuses (class, LLM role, lineage, domain). This works well when papers are well-tagged but misses:

- Papers that describe the same method with different vocabulary
- Cross-domain transfers where terminology diverges
- Papers with sparse tags (new topics, unusual framing)

**Planned upgrade:** Replace/augment tag similarity with dense vector embeddings.

| Item | Detail |
| --- | --- |
| **Where** | `src/layer2/front_detection.py` — `build_semantic_edges()` function |
| **Storage** | `paper_analyses.embedding BLOB` — column already exists in schema |
| **Model** | `text-embedding-3-small` (OpenAI) or `models/embedding-001` (Gemini) |
| **Input** | Abstract (fast, cheap) or abstract + conclusion paragraph (better quality) |
| **Similarity** | Cosine similarity over L2-normalized embeddings |
| **Threshold** | Replace `--min-similarity` IDF threshold with cosine threshold (~0.75) |
| **Cost** | ~$0.002/1000 papers (OpenAI) — negligible |
| **Integration** | Add `--use-embeddings` flag to `layer2_detect_fronts.py`; fall back to tag similarity if embedding absent |

**Implementation steps (when ready):**

1. Add `compute_and_store_embeddings(category, db)` to `src/layer2/embedding_similarity.py`
2. Modify `build_semantic_edges()` to use cosine similarity when embeddings exist
3. Add `--use-embeddings` flag to `layer2_detect_fronts.py`
4. Run embedding computation once per paper (cache in DB)
5. Test: compare front quality (tag-only vs embedding-augmented) on same snapshot

---

## Recent Updates

### 2026-02-14: Layer 3 Complete + Layer 2 Bug Fixes ✅

- ✅ Layer 3 fully implemented: data collector, graph visualizer, email renderer, email sender
- ✅ Daily/weekly/monthly email pipelines with unified CLI (`layer3_run.py`)
- ✅ Weekly email includes dedicated **Research Landscape** section from Layer 2 analysis
- ✅ 6 Layer 2 bugs fixed (code review): evolution tracking order, stale edges, force-refresh, crash-safety, no-edge fallback, weighted bridge score
- ✅ Added embedding TODO to Phase 6 (current signal: tag Jaccard; future: cosine embedding similarity)
- 🎯 **Next:** Automation & deployment (GitHub Actions, secrets, end-to-end test)

### 2026-02-12: Layer 2 Complete! ✅

- ✅ Citation graph builder with Semantic Scholar API (rate limiting, retries, pagination)
- ✅ Co-citation network construction with Salton's cosine normalization
- ✅ Louvain community detection for research fronts (configurable resolution)
- ✅ Front evolution tracking (Jaccard similarity with previous snapshots)
- ✅ Bridge paper detection with cross-front edge scoring
- ✅ LLM front summaries using Gemini 2.5 Flash
- ✅ Full CLI: `python src/scripts/layer2_detect_fronts.py`
- ✅ Switched front_summarizer config to stable `gemini-2.5-flash`
- 🎯 **Next:** Begin Layer 3 - Living reviews and email

### 2026-02-12: Layer 1 Complete! ✅

- ✅ Migrated from Anthropic to Google Gemini
- ✅ Implemented structured output with `google-genai` SDK
- ✅ Solved MAX_TOKENS issue (switched from experimental to stable model)
- ✅ Successfully tested full pipeline on real ArXiv paper
- ✅ All 4 agents working: Reader → Methods → Positioning → Database
- ✅ Created comprehensive skill document for Gemini structured output

### Key Learnings

1. **Use stable models**: `gemini-2.5-flash` (not experimental `gemini-3-flash-preview`)
2. **Structured output works**: Native JSON schema validation prevents parsing errors
3. **Set max_tokens high**: 8192+ tokens needed for complex nested schemas
4. **Monitor finish_reason**: `FinishReason.MAX_TOKENS` indicates need for more tokens
5. **Cost-effective**: ~$0.01 per paper with Gemini Flash

### What's Working

- ✅ PDF extraction from ArXiv with SHA256 caching
- ✅ 3 concurrent agents analyzing papers in parallel
- ✅ Structured JSON output with Pydantic validation
- ✅ SQLite database storage with proper JSON serialization
- ✅ CLI tool for batch processing unanalyzed papers
- ✅ Integration with existing `daily_arxiv.py` output

---

## Resources

- **Setup Guide:** [SETUP.md](SETUP.md)
- **Gemini Skill:** [docs/gemini-structured-output-skill.md](docs/gemini-structured-output-skill.md)
- **Configuration:** [src/config.py](src/config.py)
- **Database Schema:** [src/db/schema.sql](src/db/schema.sql)
