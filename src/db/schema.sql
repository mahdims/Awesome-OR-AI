-- ============================================================================
-- Research Intelligence System - Database Schema
-- ============================================================================

-- ============================================================================
-- LAYER 1: Paper Analyses
-- ============================================================================

CREATE TABLE IF NOT EXISTS paper_analyses (
    -- Primary identifiers
    arxiv_id        TEXT PRIMARY KEY,
    category        TEXT NOT NULL,  -- "LLMs for Algorithm Design" | "Generative AI for OR" | "OR for Generative AI"

    -- Paper metadata
    title           TEXT NOT NULL,
    authors         TEXT NOT NULL,      -- JSON array
    abstract        TEXT,
    published_date  DATE NOT NULL,

    -- Agent 1: Reader outputs
    affiliations    TEXT,               -- Comma-separated, sorted by prominence
    problem         TEXT NOT NULL,      -- JSON: {formal_name, short, class, properties, scale}
    methodology     TEXT NOT NULL,      -- JSON: {core_method, llm_role, llm_model_used, search_type, novelty_claim, components, training_required}
    experiments     TEXT NOT NULL,      -- JSON: {benchmarks, baselines, hardware, instance_sizes}
    results         TEXT NOT NULL,      -- JSON: {vs_baselines, scalability, statistical_rigor, limitations_acknowledged}
    artifacts       TEXT NOT NULL,      -- JSON: {code_url, models_released, new_benchmark}
    reader_confidence TEXT,              -- JSON: {problem, methodology, experiments, results, artifacts, flags}

    -- Agent 2: Methods & Connections outputs
    lineage         TEXT NOT NULL,      -- JSON: {direct_ancestors, closest_prior_work, novelty_type}
    tags            TEXT NOT NULL,      -- JSON: {methods[], problems[], contribution_type[]}
    extensions      TEXT NOT NULL,      -- JSON: {next_steps[], transferable_to[], open_weaknesses[]}
    methods_confidence TEXT,             -- JSON: {tagging_confidence, lineage_confidence, flags}

    -- Agent 3: Positioning outputs
    relevance       TEXT NOT NULL,      -- JSON: {methodological, problem, inspirational} (scores 0-10)
    significance    TEXT NOT NULL,      -- JSON: {must_read, changes_thinking, team_discussion, reasoning}
    brief           TEXT NOT NULL,      -- One-paragraph assessment

    -- Analysis metadata
    analysis_date   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_model  TEXT,               -- LLM model used (e.g., "claude-sonnet-4-5")
    pdf_hash        TEXT,               -- SHA256 of PDF (detect revisions)

    -- Semantic search support (added later)
    embedding       BLOB,               -- For similarity queries

    -- Relevance filter (Agent 3: Positioning — second filter)
    -- 1 = passes filter (max(M,P,I) >= 6 OR must_read); 0 = below threshold
    is_relevant     INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_category ON paper_analyses(category);
CREATE INDEX IF NOT EXISTS idx_date ON paper_analyses(published_date);
CREATE INDEX IF NOT EXISTS idx_tags ON paper_analyses((json_extract(tags, '$.methods')));

-- ============================================================================
-- LAYER 2: Citation Graph & Research Fronts (PER CATEGORY)
-- ============================================================================

CREATE TABLE IF NOT EXISTS citations (
    -- Directed edges in citation graph
    source_paper_id     TEXT NOT NULL,  -- Paper that cites
    target_paper_id     TEXT NOT NULL,  -- Paper being cited
    category            TEXT NOT NULL,  -- Which category's graph this belongs to
    citation_context    TEXT,           -- Optional: sentence where citation appears
    discovered_date     DATE DEFAULT (date('now')),

    PRIMARY KEY (source_paper_id, target_paper_id, category)
);

CREATE INDEX IF NOT EXISTS idx_citations_category ON citations(category);
CREATE INDEX IF NOT EXISTS idx_citations_source ON citations(source_paper_id);
CREATE INDEX IF NOT EXISTS idx_citations_target ON citations(target_paper_id);

CREATE TABLE IF NOT EXISTS citation_fetch_log (
    -- One row per corpus paper per category. Records that Semantic Scholar
    -- was queried for this paper so subsequent runs skip it and load from
    -- the citations table instead. When a new paper joins the corpus it
    -- has no row here, so only it gets fetched; old papers are reused.
    arxiv_id        TEXT NOT NULL,
    category        TEXT NOT NULL,
    fetch_date      DATE NOT NULL DEFAULT (date('now')),
    fetch_mode      TEXT NOT NULL,      -- "references" | "both"
    refs_count      INTEGER DEFAULT 0,  -- outgoing edges stored
    cited_count     INTEGER DEFAULT 0,  -- incoming edges stored (mode=both only)

    PRIMARY KEY (arxiv_id, category)
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_category ON citation_fetch_log(category);

CREATE TABLE IF NOT EXISTS cocitation_edges (
    -- Undirected co-citation network (per category)
    paper1_id       TEXT NOT NULL,
    paper2_id       TEXT NOT NULL,
    category        TEXT NOT NULL,
    cocitation_count INTEGER NOT NULL,  -- How many papers cite both
    strength        REAL,               -- Normalized weight [0,1]
    snapshot_date   DATE NOT NULL,

    PRIMARY KEY (paper1_id, paper2_id, category, snapshot_date),
    CHECK (paper1_id < paper2_id)  -- Enforce ordering to avoid duplicates
);

CREATE INDEX IF NOT EXISTS idx_cocitation_category ON cocitation_edges(category, snapshot_date);

CREATE TABLE IF NOT EXISTS research_fronts (
    -- Detected communities in co-citation network
    front_id            TEXT NOT NULL,  -- Format: "{category_slug}_{snapshot_date}_front_{n}"
    category            TEXT NOT NULL,
    snapshot_date       DATE NOT NULL,

    -- Front composition
    core_papers         TEXT NOT NULL,  -- JSON array of paper IDs
    size                INTEGER NOT NULL,
    internal_density    REAL,           -- Edge density within community

    -- Semantic characterization (from Layer 1 tags)
    dominant_methods    TEXT,           -- JSON array of top methods
    dominant_problems   TEXT,           -- JSON array of top problems

    -- Evolution metrics (compared to previous snapshot)
    growth_rate         REAL,           -- % change in size
    stability           REAL,           -- Jaccard similarity with previous version
    status              TEXT,           -- "emerging" | "growing" | "stable" | "declining" | "merged" | "split"

    -- LLM-generated enrichment
    name                TEXT,           -- 6-10 word human-readable title for the front
    summary             TEXT,           -- 3 paragraph narrative (theme, contributions, trajectory)
    future_directions   TEXT,           -- JSON array of 3-5 concrete research directions

    PRIMARY KEY (front_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_fronts_category ON research_fronts(category, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_fronts_status ON research_fronts(status);

CREATE TABLE IF NOT EXISTS front_lineage (
    -- Tracks front evolution across snapshots
    current_front_id    TEXT NOT NULL,
    previous_front_id   TEXT,
    category            TEXT NOT NULL,
    snapshot_date       DATE NOT NULL,
    relationship        TEXT NOT NULL,  -- "continuation" | "merge_from" | "split_from" | "new"
    overlap_score       REAL,           -- Jaccard similarity

    PRIMARY KEY (current_front_id, previous_front_id, snapshot_date),
    FOREIGN KEY (current_front_id) REFERENCES research_fronts(front_id)
);

CREATE TABLE IF NOT EXISTS bridge_papers (
    -- Papers connecting multiple fronts WITHIN a category
    paper_id            TEXT NOT NULL,
    category            TEXT NOT NULL,
    snapshot_date       DATE NOT NULL,
    home_front_id       TEXT NOT NULL,  -- Primary front
    connected_fronts    TEXT NOT NULL,  -- JSON array of connected front IDs
    bridge_score        REAL NOT NULL,  -- [0,1]: fraction of cross-front links

    PRIMARY KEY (paper_id, category, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_bridge_category ON bridge_papers(category, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_bridge_score ON bridge_papers(bridge_score DESC);

-- ============================================================================
-- LAYER 3: Living Review Metadata
-- ============================================================================

CREATE TABLE IF NOT EXISTS review_updates (
    -- Audit trail of living review changes
    update_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    category        TEXT NOT NULL,
    update_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_type     TEXT NOT NULL,      -- "daily" | "weekly" | "monthly"
    papers_added    INTEGER DEFAULT 0,
    fronts_changed  INTEGER DEFAULT 0,
    commit_hash     TEXT,               -- Git commit SHA
    summary         TEXT                -- Brief description of changes
);

CREATE INDEX IF NOT EXISTS idx_updates_category ON review_updates(category, update_date);
