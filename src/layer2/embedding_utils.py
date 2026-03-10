"""
Embedding generation and storage utilities for Layer 2 semantic edges.

Generates OpenAI embeddings for papers lazily (only when embedding IS NULL in DB),
stores them as float32 bytes in the existing `embedding BLOB` column, and
provides a cosine-ready unit-vector dict for use in build_semantic_edges.

Abstract is required — papers without one are excluded and a warning is printed.
Title-only fallback is intentionally omitted: in this narrow domain (LLM + OR),
titles follow near-identical templates ("LLM-guided X for Y optimization"),
producing noisy vectors that would create spurious edges.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 8000  # API limit guard — not a tuning param, kept hardcoded


def _load_embedding_config() -> dict:
    """Load layer2.embedding section from research_config/model_config.yaml.

    Returns the raw dict so callers can use .get() with their own defaults.
    Falls back to {} on any error (missing file, bad YAML, import failure).
    """
    try:
        import yaml
        config_path = Path(__file__).parent.parent.parent / "research_config" / "model_config.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg.get("layer2", {}).get("embedding", {})
    except Exception:
        return {}


_emb_cfg = _load_embedding_config()

# Tuning parameters — read from model_config.yaml (layer2.embedding section).
# Hardcoded values are fallbacks when the key is absent.
# All three are independently tunable and have distinct meanings:
EMBEDDING_MODEL  = _emb_cfg.get("model_name", "text-embedding-3-large")
BATCH_SIZE       = int(_emb_cfg.get("batch_size", 100))
# cosine_threshold: cosine below this → emb_score = 0 (removes ~0.65 domain baseline)
SIM_THRESHOLD    = float(_emb_cfg.get("cosine_threshold", 0.70))
# embedding_weight: embedding's share of the 0-1 score vs tag signals (0=tags only, 1=embedding only)
EMB_WEIGHT       = float(_emb_cfg.get("embedding_weight", 0.70))
# semantic_edge_weight: how much all semantic edges count vs citation edges in Louvain
SEMANTIC_WEIGHT  = float(_emb_cfg.get("semantic_edge_weight", 1.0))


def _backfill_abstracts(paper_ids: List[str], db) -> None:
    """
    Fetch abstracts from the ArXiv Atom API for papers that have none in the DB.

    Uses the same ArXiv API pattern as Layer 0. Stores results back into
    paper_analyses.abstract so subsequent runs skip the API entirely.
    Silently skips on any network/parse failure.
    """
    import requests
    import xml.etree.ElementTree as ET

    # Only fetch papers that are actually missing abstracts
    need = []
    for pid in paper_ids:
        row = db.get_analysis(pid)
        if row and not (row.get("abstract") or "").strip():
            need.append(pid)

    if not need:
        return

    print(f"  Fetching {len(need)} missing abstracts from ArXiv API...")

    ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
    BATCH = 50  # ArXiv API handles up to ~100 but 50 is safe
    fetched = 0

    for start in range(0, len(need), BATCH):
        batch = need[start: start + BATCH]
        id_list = ",".join(batch)
        url = f"http://export.arxiv.org/api/query?id_list={id_list}&max_results={len(batch)}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for entry in root.findall("atom:entry", ARXIV_NS):
                # Extract arxiv_id from the <id> element
                id_el = entry.find("atom:id", ARXIV_NS)
                if id_el is None or not id_el.text:
                    continue
                # ID is like "http://arxiv.org/abs/2401.02051v1" — extract just the base ID
                raw_id = id_el.text.strip().split("/abs/")[-1]
                arxiv_id = raw_id.split("v")[0]  # strip version suffix

                summary_el = entry.find("atom:summary", ARXIV_NS)
                if summary_el is not None and summary_el.text:
                    abstract = summary_el.text.replace("\n", " ").strip()
                    db.store_abstract(arxiv_id, abstract)
                    fetched += 1
        except Exception as e:
            logger.warning(f"ArXiv abstract fetch failed for batch [{start}:{start+len(batch)}]: {e}")

    print(f"  Abstracts: {fetched}/{len(need)} fetched and stored")


def load_or_generate_embeddings(
    corpus_ids: List[str],
    db,
) -> Dict[str, "np.ndarray"]:
    """
    Return {arxiv_id: unit_vector (float32)} for all corpus papers with abstracts.

    Phases:
      1. Load existing embeddings from DB (embedding IS NOT NULL → free)
      2. For papers with NULL embedding, fetch title+abstract, skip if no abstract
      3. Call OpenAI API in batches for embeddable papers
      4. Store results in DB, return full vector dict

    Papers without abstracts are excluded with a printed warning.
    If OPENAI_API_KEY is not set or openai/numpy are unavailable, returns {}.
    """
    if not _NUMPY_AVAILABLE or not _OPENAI_AVAILABLE:
        logger.warning(
            "Embedding signal skipped: numpy or openai not available. "
            "Install with: pip install openai numpy"
        )
        return {}

    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("Embedding signal skipped: OPENAI_API_KEY not set.")
        return {}

    result: Dict[str, "np.ndarray"] = {}
    missing_ids: List[str] = []

    # Phase 1: Load existing embeddings from DB
    for pid in corpus_ids:
        blob = db.get_embedding(pid)
        if blob is not None:
            result[pid] = np.frombuffer(blob, dtype=np.float32)
        else:
            missing_ids.append(pid)

    if not missing_ids:
        return result

    # Phase 1b: Backfill abstracts from ArXiv API for papers that have none
    _backfill_abstracts(missing_ids, db)

    # Phase 2: Build embed texts — abstract required, no title-only fallback
    no_abstract_ids: List[str] = []
    embeddable_ids: List[str] = []
    texts: List[str] = []

    for pid in missing_ids:
        row = db.get_analysis(pid)
        if not row:
            continue
        abstract = (row.get("abstract") or "").strip()
        if not abstract:
            no_abstract_ids.append(pid)
            continue
        title = (row.get("title") or "").strip()
        text = f"{title}. {abstract}" if title else abstract
        texts.append(text[:MAX_TEXT_CHARS])
        embeddable_ids.append(pid)

    if no_abstract_ids:
        print(f"  [WARN] {len(no_abstract_ids)} papers excluded from Signal 10 "
              f"(no abstract): {no_abstract_ids}")

    if not texts:
        return result

    print(f"  Generating embeddings for {len(texts)} papers "
          f"(model={EMBEDDING_MODEL})...")

    # Phase 3: Batch API calls
    vectors = _batch_embed(texts)

    # Phase 4: Store and collect
    stored = 0
    for pid, vec in zip(embeddable_ids, vectors):
        if vec is None:
            continue
        db.store_embedding(pid, vec.astype(np.float32).tobytes())
        result[pid] = vec
        stored += 1

    print(f"  Embeddings: {stored} generated+stored, "
          f"{len(result) - stored} loaded from cache, "
          f"{len(no_abstract_ids)} skipped (no abstract)")
    return result


def _batch_embed(texts: List[str]) -> List[Optional["np.ndarray"]]:
    """
    Call OpenAI embeddings API in batches of BATCH_SIZE.

    Returns a list parallel to `texts` of unit-norm float32 arrays.
    Failed batches yield None sentinels so the caller can skip them
    without aborting the rest of the pipeline.
    """
    client = OpenAI()
    results: List[Optional["np.ndarray"]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start: start + BATCH_SIZE]
        try:
            response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            # Sort by index to guarantee order matches input
            for item in sorted(response.data, key=lambda x: x.index):
                vec = np.array(item.embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                results.append(vec)
        except Exception as e:
            logger.warning(
                f"Embedding API call failed for batch [{start}:{start+len(batch)}]: {e}"
            )
            results.extend([None] * len(batch))

    return results
