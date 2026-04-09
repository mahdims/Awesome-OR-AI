"""
Layer 2: Co-citation Network & Research Front Detection

Builds co-citation networks from citation graphs and applies
Louvain community detection to identify research fronts.
"""

import json
import statistics
from collections import Counter, defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

try:
    import pycombo
except ImportError:
    pycombo = None
    print("[WARN] pycombo not installed. Install with: pip install pycombo")

from db.database import Database


def _parse_analysis_fields(analysis: dict) -> tuple:
    """
    Parse a paper_analyses row into clustering-relevant semantic fields.

    Returns: (methods, problems, class_, llm_role, search_type, benchmarks, props,
              ancestors, framework_lineage, specific_domain, llm_coupling)
    """
    def _j(val, default):
        if val is None:
            return default
        if isinstance(val, (dict, list)):
            return val
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return default

    tags     = _j(analysis.get('tags'), {})
    methods  = set(tags.get('methods', []))
    problems = set(tags.get('problems', []))
    application      = set(tags.get('application', []))       # new: real-world sector
    problem_properties = set(tags.get('problem_properties', []))  # new: cross-cutting characteristics

    # Fine-grained open-ended fields (added by improved methods.txt)
    framework_lineage = tags.get('framework_lineage') or ''
    specific_domain   = tags.get('specific_domain') or ''
    llm_coupling      = tags.get('llm_coupling') or ''

    problem = _j(analysis.get('problem'), {})
    # Key is "class_" not "class": model_dump() is called without by_alias in schemas.py
    class_  = problem.get('class_', '')
    props   = set(problem.get('properties', []))

    meth        = _j(analysis.get('methodology'), {})
    llm_role    = meth.get('llm_role', '')    # "heuristic_generator" | "code_writer" | ...
    search_type = meth.get('search_type', '') # "constructive" | "improvement" | "hybrid"

    exp        = _j(analysis.get('experiments'), {})
    benchmarks = set(exp.get('benchmarks', []))

    lineage   = _j(analysis.get('lineage'), {})
    ancestors = lineage.get('direct_ancestors', [])  # list of {paper, relationship}

    return (methods, problems, class_, llm_role, search_type, benchmarks, props,
            ancestors, framework_lineage, specific_domain, llm_coupling,
            application, problem_properties)


def build_semantic_edges(corpus_ids: set, db, semantic_weight: float,
                          G: nx.Graph,
                          min_similarity: float = 0.3) -> None:
    """
    Inject IDF-weighted semantic similarity edges into graph G (in-place).

    Addresses the near-complete-graph problem in narrow domains: a tag shared
    by 40/44 papers (e.g. 'evolution_of_heuristics') carries almost no
    information about which papers are MOST similar, whereas a tag shared by
    only 3/44 papers is a strong "same cluster" signal.

    IDF-weighted Jaccard:
        idf(t)  = log(N / df(t))       — high for rare tags, near-zero for common tags
        sim(A,B) = Σ idf(t∩B) / Σ idf(t∈A∪B)

    Additional IDF-scaled categorical bonuses (zero if all papers share the
    same value, which is noise-free):
        class_bonus  = idf(class_)  * 0.30   if cls_a == cls_b
        role_bonus   = idf(role)    * 0.25   if role_a == role_b
        stype_bonus  = idf(stype)   * 0.15   if stype_a == stype_b

    Plus IDF-weighted benchmark Jaccard (bench tags are usually specific,
    so naturally high IDF).

    Edges are only created when total weighted similarity >= min_similarity,
    preventing near-complete connectivity in homogeneous corpora.
    """
    from math import log as _log

    corpus_list = sorted(corpus_ids)
    analyses = {}
    missing = 0
    for pid in corpus_list:
        row = db.get_analysis(pid)
        if row is None:
            missing += 1
            continue
        analyses[pid] = _parse_analysis_fields(row)

    print(f"  Layer 1 data: {len(analyses)}/{len(corpus_list)} papers found "
          f"({missing} skipped — no analysis row)")
    if not analyses:
        print("  [WARN] No Layer 1 analyses found. Semantic edges skipped.")
        return

    N = len(analyses)

    # --- Build IDF for all tags (methods + problems) and benchmarks ---
    tag_df = Counter()      # document frequency per tag
    bench_df = Counter()    # document frequency per benchmark name
    class_df = Counter()    # document frequency per problem class value
    role_df = Counter()     # document frequency per llm_role value
    stype_df = Counter()    # document frequency per search_type value
    lineage_df = Counter()  # document frequency per framework_lineage value
    domain_df = Counter()   # document frequency per specific_domain value
    coupling_df = Counter() # document frequency per llm_coupling value

    for pid, (methods, problems, cls, role, stype, benchmarks, _props,
              _anc, fw_lineage, sp_domain, llm_cpl, app, pp) in analyses.items():
        for t in methods | problems | app | pp:
            tag_df[t] += 1
        for b in benchmarks:
            bench_df[b] += 1
        if cls:
            class_df[cls] += 1
        if role:
            role_df[role] += 1
        if stype:
            stype_df[stype] += 1
        if fw_lineage:
            lineage_df[fw_lineage] += 1
        if sp_domain:
            domain_df[sp_domain] += 1
        if llm_cpl:
            coupling_df[llm_cpl] += 1

    def idf(df_count):
        """IDF score: 0 when df=N (appears in all papers), high when df is small."""
        if df_count <= 0:
            return 0.0
        return _log(N / df_count)

    def idf_jaccard(tags_a: set, tags_b: set, df_map: Counter) -> float:
        """IDF-weighted Jaccard similarity for a set of discrete labels."""
        union = tags_a | tags_b
        if not union:
            return 0.0
        num = sum(idf(df_map[t]) for t in tags_a & tags_b)
        den = sum(idf(df_map[t]) for t in union)
        return num / den if den > 0 else 0.0

    # citation_scale: mean co-citation edge weight before semantic edges are added.
    # A co-citation edge weight = number of external papers that cited both papers together.
    # This anchors semantic edge weights to the same numerical scale as citation edges,
    # so Louvain treats them proportionally.
    # semantic_weight (caller param) then scales all semantic edges up/down relative
    # to citation evidence (default 1.0 = equal footing with average citation pair).
    citation_scale = (sum(d['weight'] for _, _, d in G.edges(data=True)) / G.number_of_edges()
                      if G.number_of_edges() > 0 else 1.0)

    # --- Load / generate OpenAI embeddings ---
    # Lazy: generates only for papers with embedding IS NULL in DB (stores result).
    # Returns {} silently if OPENAI_API_KEY not set — embedding score degrades to 0.
    # Fallback constants used when embedding_utils cannot be imported.
    SIM_THRESHOLD = 0.70
    EMB_WEIGHT    = 0.70
    emb_vecs: dict = {}
    if _NUMPY_AVAILABLE:
        try:
            from layer2.embedding_utils import load_or_generate_embeddings, SIM_THRESHOLD, EMB_WEIGHT
            emb_vecs = load_or_generate_embeddings(corpus_list, db)
        except Exception as e:
            print(f"  [WARN] Embedding signal unavailable: {e}")

    # log_N: normalizer for categorical IDF scores → maps idf(df) ∈ [0, log(N)] to [0, 1]
    log_N = _log(N) if N > 1 else 1.0

    # Print the active configuration so the user can see exactly what is driving edges
    emb_status = (f"{len(emb_vecs)} embeddings loaded"
                  if emb_vecs else "disabled (OPENAI_API_KEY not set or unavailable)")
    print(f"  Embedding signal: {emb_status}")
    print(f"  Config: cosine_threshold={SIM_THRESHOLD}, embedding_weight={EMB_WEIGHT}, "
          f"min_score={min_similarity}, semantic_edge_weight={semantic_weight:.2g}, "
          f"citation_scale={citation_scale:.2f}")

    ids = sorted(analyses.keys())
    tag_new = lineage_new = 0
    emb_driven = 0    # new edges where embedding was the primary driver (has_emb=True)
    tag_only_new = 0  # new edges created without embeddings (tag-only path)
    n_emb_pairs = 0   # total pairs where both papers have embeddings
    all_scores: List[float] = []     # all pairwise scores for distribution reporting
    all_cosines: List[float] = []    # raw cosine values for all emb pairs (before threshold)
    all_tag_scores: List[float] = [] # tag scores for all pairs

    for i in range(len(ids)):
        pid_a = ids[i]
        m_a, p_a, cls_a, role_a, stype_a, bench_a, _p, _anc_a, lin_a, dom_a, cpl_a, app_a, pp_a = analyses[pid_a]
        tags_a = m_a | p_a | app_a | pp_a

        for j in range(i + 1, len(ids)):
            pid_b = ids[j]
            m_b, p_b, cls_b, role_b, stype_b, bench_b, _p, _anc_b, lin_b, dom_b, cpl_b, app_b, pp_b = analyses[pid_b]
            tags_b = m_b | p_b | app_b | pp_b

            # --- All signals normalized to [0, 1] ---

            # Tag signal: IDF-weighted Jaccard over methods + problems  [0, 1] naturally
            s_tags = idf_jaccard(tags_a, tags_b, tag_df)

            # Benchmark signal: IDF-weighted Jaccard over benchmark names  [0, 1] naturally
            s_bench = idf_jaccard(bench_a, bench_b, bench_df)

            # Categorical signals: each normalized by log(N) so idf ∈ [0, log(N)] → [0, 1]
            # Returns 0 when unmatched or when all papers share the same value (IDF≈0)
            def _nc(val_a, val_b, df_map):
                if not val_a or val_a != val_b:
                    return 0.0
                return idf(df_map[val_a]) / log_N

            # s_cat weights: framework_lineage and llm_coupling raised after improved
            # tag taxonomy (more discriminating values); reader-derived cls/role/stype lowered.
            s_cat = (0.20 * _nc(cls_a,   cls_b,   class_df)    # problem class     (was 0.25)
                   + 0.15 * _nc(role_a,  role_b,  role_df)     # LLM role          (was 0.20)
                   + 0.10 * _nc(stype_a, stype_b, stype_df)    # search type       (was 0.15)
                   + 0.30 * _nc(lin_a,   lin_b,   lineage_df)  # framework lineage (was 0.20)
                   + 0.15 * _nc(dom_a,   dom_b,   domain_df)   # specific domain
                   + 0.10 * _nc(cpl_a,   cpl_b,   coupling_df) # LLM coupling      (was 0.05)
                   )  # s_cat ∈ [0, 1] since weights sum to 1.0

            # Tag score: s_tags raised to 0.60 (now includes application + problem_properties
            # in addition to methods + problems); s_bench reduced since tags compensate.
            tag_score = min(1.0,
                0.60 * s_tags    # methods/problems/application/properties (was 0.50)
              + 0.30 * s_cat     # categorical structure
              + 0.10 * s_bench   # benchmark overlap: reduced (was 0.20)
            )

            # Embedding score: normalized cosine  [0, 1]
            # 0 when cos ≤ SIM_THRESHOLD, 1 when cos = 1.0
            # SIM_THRESHOLD removes the ~0.65 domain baseline of text-embedding-3-large
            emb_score = 0.0
            all_tag_scores.append(tag_score)
            vec_a = emb_vecs.get(pid_a) if emb_vecs else None
            vec_b = emb_vecs.get(pid_b) if emb_vecs else None
            has_emb = vec_a is not None and vec_b is not None
            if has_emb:
                cos = float(np.dot(vec_a, vec_b))
                all_cosines.append(cos)        # raw cosine before threshold (for diagnostics)
                if cos > SIM_THRESHOLD:
                    emb_score = (cos - SIM_THRESHOLD) / (1.0 - SIM_THRESHOLD)

            # Final score [0, 1]:
            # With embeddings: EMB_WEIGHT fraction from embedding, rest from tags
            # Without embeddings for this pair: tag-only (graceful degradation)
            if has_emb:
                n_emb_pairs += 1
                score = EMB_WEIGHT * emb_score + (1.0 - EMB_WEIGHT) * tag_score
            else:
                score = tag_score

            # Collect ALL scores for distribution reporting (before threshold filter)
            all_scores.append(score)

            if score < min_similarity:
                continue

            # Edge weight: score × citation_scale × semantic_weight
            # citation_scale keeps semantic edges on the same numerical scale as citation edges
            # semantic_weight is the user's trust factor (--semantic-weight CLI arg)
            w = score * citation_scale * semantic_weight
            if G.has_edge(pid_a, pid_b):
                G[pid_a][pid_b]['weight'] += w
            else:
                G.add_edge(pid_a, pid_b, weight=w)
                tag_new += 1
                if has_emb:
                    emb_driven += 1
                else:
                    tag_only_new += 1

    # Lineage edges (only exact arxiv_id matches in corpus)
    # ancestors is index 7 in the 11-field tuple
    for pid_a, (_m, _p, _cls, _role, _stype, _bench, _props,
                ancestors, _lin, _dom, _cpl, _app, _pp) in analyses.items():
        for entry in ancestors:
            anc = entry.get('paper', '')
            if anc not in corpus_ids:
                continue  # title-only refs or out-of-corpus papers: skip silently
            w = citation_scale * semantic_weight
            if G.has_edge(pid_a, anc):
                G[pid_a][anc]['weight'] += w
            else:
                G.add_edge(pid_a, anc, weight=w)
                lineage_new += 1

    total_pairs = len(all_scores)
    above = [s for s in all_scores if s >= min_similarity]
    print(f"  Pairwise scoring: {total_pairs} pairs evaluated "
          f"({n_emb_pairs} with embeddings, {total_pairs - n_emb_pairs} tag-only)")
    print(f"  Pairwise edges added: {tag_new} "
          f"({emb_driven} embedding-driven, {tag_only_new} tag-only) | "
          f"{lineage_new} lineage edges")
    if all_scores:
        std_str = f"{statistics.stdev(all_scores):.3f}" if len(all_scores) > 1 else "n/a"
        print(f"  Score distribution (ALL {total_pairs} pairs): "
              f"min={min(all_scores):.3f}, max={max(all_scores):.3f}, "
              f"mean={statistics.mean(all_scores):.3f}, "
              f"median={statistics.median(all_scores):.3f}, "
              f"std={std_str} | "
              f"above threshold ({min_similarity}): {len(above)}/{total_pairs}")
    if all_cosines:
        thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        thresh_str = " | ".join(
            f"≥{t}: {sum(1 for c in all_cosines if c >= t)}/{len(all_cosines)}"
            for t in thresholds
        )
        cos_std = f"{statistics.stdev(all_cosines):.3f}" if len(all_cosines) > 1 else "n/a"
        print(f"  Raw cosine ({len(all_cosines)} emb pairs): "
              f"min={min(all_cosines):.3f}, max={max(all_cosines):.3f}, "
              f"mean={statistics.mean(all_cosines):.3f}, "
              f"median={statistics.median(all_cosines):.3f}, "
              f"std={cos_std}")
        print(f"  Cosine coverage: {thresh_str}")
    if all_tag_scores:
        print(f"  Tag score ({len(all_tag_scores)} pairs): "
              f"min={min(all_tag_scores):.3f}, max={max(all_tag_scores):.3f}, "
              f"mean={statistics.mean(all_tag_scores):.3f}, "
              f"median={statistics.median(all_tag_scores):.3f}")


def build_cocitation_network(citation_graph: nx.DiGraph,
                              corpus_ids: set,
                              db=None,
                              semantic_weight: float = 1.0,
                              bib_coupling_factor: float = 0.0,
                              min_similarity: float = 0.9) -> nx.Graph:
    """
    Build a weighted network combining co-citation, bibliographic coupling, and
    optional semantic similarity edges for Louvain community detection.

    Signal hierarchy (by default):
      - Co-citation (full weight): two corpus papers cited together by an external paper
      - Bibliographic coupling (bib_coupling_factor × weight): shared references — DISABLED
        by default (factor=0.0) because in narrow domains ALL papers cite the same
        foundational work, creating a near-complete graph that obscures true clustering
      - Semantic similarity (semantic_weight, PRIMARY): IDF-weighted Layer 1 analysis
        data (tag Jaccard, LLM role match, benchmark Jaccard, lineage, problem class, domain)

    Args:
        citation_graph:      Directed graph (A->B means A cites B)
        corpus_ids:          Set of paper IDs in our corpus
        db:                  Connected Database instance (needed for semantic edges)
        semantic_weight:     Scale for semantic similarity edges (default 1.0 = primary)
        bib_coupling_factor: Scale for bibliographic coupling (default 0.0 = disabled)
        min_similarity:      Min IDF-weighted score to create a semantic edge (default 0.9)

    Returns:
        Undirected weighted graph ready for Louvain clustering
    """
    # --- Step 1: Co-citation edges ---
    # For each citing paper, collect which corpus papers it references
    citing_to_refs = defaultdict(set)
    for src, tgt in citation_graph.edges():
        if tgt in corpus_ids:
            citing_to_refs[src].add(tgt)

    cocitation_counts = Counter()
    for _citing_paper, refs in citing_to_refs.items():
        refs_list = sorted(refs)
        for i in range(len(refs_list)):
            for j in range(i + 1, len(refs_list)):
                cocitation_counts[(refs_list[i], refs_list[j])] += 1

    G = nx.Graph()
    for paper_id in corpus_ids:
        G.add_node(paper_id)

    for (p1, p2), count in cocitation_counts.items():
        G.add_edge(p1, p2, weight=count)

    cocit_edge_count = G.number_of_edges()
    if cocitation_counts:
        cc_vals = list(cocitation_counts.values())
        cc_std = f"{statistics.stdev(cc_vals):.2f}" if len(cc_vals) > 1 else "n/a"
        print(f"  Co-citation edges: {cocit_edge_count} "
              f"(count range: {min(cc_vals)}–{max(cc_vals)}, "
              f"mean={statistics.mean(cc_vals):.2f}, "
              f"median={statistics.median(cc_vals):.1f}, "
              f"std={cc_std}, "
              f"cited ≥2x by same paper: {sum(1 for v in cc_vals if v >= 2)})")
    else:
        print(f"  Co-citation edges: {cocit_edge_count}")

    # --- Step 2: Bibliographic coupling edges (skip if factor=0.0) ---
    # Two corpus papers are coupled if they share at least one reference.
    bib_new = 0
    bib_merged = 0
    if bib_coupling_factor > 0.0:
        paper_refs = {pid: set(citation_graph.successors(pid)) for pid in corpus_ids}
        corpus_list = sorted(corpus_ids)
        for i in range(len(corpus_list)):
            for j in range(i + 1, len(corpus_list)):
                shared = paper_refs.get(corpus_list[i], set()) & paper_refs.get(corpus_list[j], set())
                if not shared:
                    continue
                if G.has_edge(corpus_list[i], corpus_list[j]):
                    G[corpus_list[i]][corpus_list[j]]['weight'] += len(shared) * bib_coupling_factor
                    bib_merged += 1
                else:
                    G.add_edge(corpus_list[i], corpus_list[j],
                               weight=len(shared) * bib_coupling_factor)
                    bib_new += 1

    print(f"  Bibliographic coupling: {bib_new} new edges, "
          f"{bib_merged} merged with co-citation edges (factor={bib_coupling_factor})")

    # --- Step 3: Semantic edges (primary signal when db provided) ---
    if db is not None and semantic_weight > 0.0:
        build_semantic_edges(corpus_ids, db, semantic_weight, G, min_similarity)

    return G


def normalize_cocitation_weights(G: nx.Graph) -> nx.Graph:
    """
    Normalize co-citation weights to [0, 1] range.

    Uses Salton's cosine normalization:
        strength(i,j) = cocitation(i,j) / sqrt(citations(i) * citations(j))
    """
    if G.number_of_edges() == 0:
        return G

    max_weight = max(d['weight'] for _, _, d in G.edges(data=True))
    if max_weight == 0:
        return G

    for u, v, d in G.edges(data=True):
        d['strength'] = d['weight'] / max_weight

    return G


def detect_fronts(cocitation_graph: nx.Graph,
                  category: str,
                  snapshot_date: Optional[str] = None,
                  min_front_size: int = 2,
                  resolution: float = 1.0) -> List[Dict]:
    """
    Detect research fronts using Louvain community detection.

    Args:
        cocitation_graph: Undirected weighted co-citation graph
        category: Category name
        snapshot_date: Date string (YYYY-MM-DD), defaults to today
        min_front_size: Minimum papers in a front to keep it
        resolution: Louvain resolution parameter (higher = more communities)

    Returns:
        List of front dicts ready for database insertion
    """
    if pycombo is None:
        raise ImportError("pycombo required: pip install pycombo")

    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    # Create category slug for front IDs
    slug = category.lower().replace(" ", "_").replace("/", "_")[:20]

    # Louvain requires at least one edge; without edges there is no clustering
    # signal — returning an empty list is more honest than one artificial front.
    if cocitation_graph.number_of_edges() == 0:
        print("[WARN] No co-citation edges found. Cannot detect fronts. "
              "Try --min-similarity lower, --bib-coupling-factor 0.3, or "
              "run layer2_detect_fronts.py with more analyzed papers.")
        return []

    # Run Combo community detection
    partition, _ = pycombo.execute(
        cocitation_graph,
        weight='weight',
        modularity_resolution=resolution,
        random_seed=42
    )

    # Group papers by community
    communities = defaultdict(list)
    for paper_id, comm_id in partition.items():
        communities[comm_id].append(paper_id)

    # Build front records
    fronts = []
    for comm_id, papers in sorted(communities.items(), key=lambda x: -len(x[1])):
        if len(papers) < min_front_size:
            continue

        # Calculate internal density
        subgraph = cocitation_graph.subgraph(papers)
        n = len(papers)
        possible_edges = n * (n - 1) / 2
        density = subgraph.number_of_edges() / possible_edges if possible_edges > 0 else 0

        front_id = f"{slug}_{snapshot_date}_front_{comm_id}"

        fronts.append({
            'front_id': front_id,
            'category': category,
            'snapshot_date': snapshot_date,
            'core_papers': papers,
            'size': len(papers),
            'internal_density': round(density, 4),
            'dominant_methods': [],  # Filled by enrich_fronts_with_tags
            'dominant_problems': [],
            'status': 'new',
        })

    print(f"  Detected {len(fronts)} fronts (min size={min_front_size})")
    for f in fronts:
        print(f"    {f['front_id']}: {f['size']} papers, density={f['internal_density']:.3f}")

    return fronts


def enrich_fronts_with_tags(fronts: List[Dict], db: Database) -> List[Dict]:
    """
    Enrich fronts with dominant methods/problems from Layer 1 analyses.

    Aggregates method and problem tags from all papers in each front.
    """
    for front in fronts:
        method_counts = Counter()
        problem_counts = Counter()

        for paper_id in front['core_papers']:
            analysis = db.get_analysis(paper_id)
            if not analysis:
                continue

            # Parse tags JSON
            tags = analysis.get('tags', '{}')
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except json.JSONDecodeError:
                    continue

            for method in tags.get('methods', []):
                method_counts[method] += 1
            for problem in tags.get('problems', []):
                problem_counts[problem] += 1

        # Take top 5 methods and problems
        front['dominant_methods'] = [m for m, _ in method_counts.most_common(5)]
        front['dominant_problems'] = [p for p, _ in problem_counts.most_common(5)]

    return fronts


def compare_with_previous(fronts: List[Dict], category: str,
                           db: Database) -> List[Dict]:
    """
    Compare current fronts with previous snapshot to track evolution.

    Sets status and stability fields based on Jaccard similarity
    with the most recent prior snapshot.
    """
    previous_fronts = db.get_latest_fronts(category)

    if not previous_fronts:
        # First snapshot - all fronts are "new"
        for f in fronts:
            f['growth_rate'] = 0.0
            f['stability'] = 0.0
            f['status'] = 'new'
        return fronts

    # Build paper sets for previous fronts
    prev_paper_sets = {}
    for pf in previous_fronts:
        papers = pf.get('core_papers', '[]')
        if isinstance(papers, str):
            papers = json.loads(papers)
        prev_paper_sets[pf['front_id']] = set(papers)

    # Match current fronts to previous ones by max Jaccard similarity
    for front in fronts:
        current_set = set(front['core_papers'])
        best_jaccard = 0.0
        best_prev_id = None
        best_prev_size = 0

        for prev_id, prev_set in prev_paper_sets.items():
            intersection = len(current_set & prev_set)
            union = len(current_set | prev_set)
            jaccard = intersection / union if union > 0 else 0
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_prev_id = prev_id
                best_prev_size = len(prev_set)

        front['stability'] = round(best_jaccard, 4)

        if best_prev_id and best_jaccard > 0.3:
            # This front matches a previous one
            growth = (front['size'] - best_prev_size) / best_prev_size if best_prev_size > 0 else 0
            front['growth_rate'] = round(growth, 4)

            if best_jaccard > 0.8:
                front['status'] = 'stable'
            elif growth > 0.2:
                front['status'] = 'growing'
            elif growth < -0.2:
                front['status'] = 'declining'
            else:
                front['status'] = 'stable'
        else:
            front['growth_rate'] = 0.0
            front['status'] = 'emerging'

    return fronts


def store_cocitation_edges(cocitation_graph: nx.Graph, category: str,
                            snapshot_date: str, db: Database):
    """Store co-citation edges in the database."""
    for u, v, d in cocitation_graph.edges(data=True):
        # Enforce ordering (paper1_id < paper2_id)
        p1, p2 = (u, v) if u < v else (v, u)
        db.execute(
            """INSERT OR REPLACE INTO cocitation_edges
               (paper1_id, paper2_id, category, cocitation_count, strength, snapshot_date)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (p1, p2, category, d.get('weight', 1),
             d.get('strength', 0.0), snapshot_date)
        )
    db.commit()


def run_front_detection(category: str,
                         citation_graph: nx.DiGraph,
                         snapshot_date: Optional[str] = None,
                         min_front_size: int = 2,
                         resolution: float = 1.0) -> List[Dict]:
    """
    Full front detection pipeline for a category.

    1. Build co-citation network from citation graph
    2. Normalize weights
    3. Detect communities (Louvain)
    4. Enrich with Layer 1 tags
    5. Compare with previous snapshot
    6. Store results

    Returns list of detected fronts.
    """
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    db = Database()

    # Get corpus paper IDs
    with db:
        papers = db.get_papers_by_category(category)
    corpus_ids = {p['arxiv_id'] for p in papers}

    print(f"\n  Building co-citation network...")
    cocitation = build_cocitation_network(citation_graph, corpus_ids)
    cocitation = normalize_cocitation_weights(cocitation)

    print(f"  Co-citation network: {cocitation.number_of_nodes()} nodes, "
          f"{cocitation.number_of_edges()} edges")

    # Store co-citation edges
    with db:
        store_cocitation_edges(cocitation, category, snapshot_date, db)

    # Detect fronts
    print(f"\n  Running Combo community detection...")
    fronts = detect_fronts(cocitation, category, snapshot_date,
                           min_front_size, resolution)

    if not fronts:
        print("  [INFO] No fronts detected")
        return []

    # Enrich with tags from Layer 1
    print(f"\n  Enriching fronts with Layer 1 tags...")
    with db:
        fronts = enrich_fronts_with_tags(fronts, db)

    # Compare with previous snapshot
    print(f"\n  Comparing with previous snapshot...")
    with db:
        fronts = compare_with_previous(fronts, category, db)

    # Store fronts in database
    print(f"\n  Storing {len(fronts)} fronts in database...")
    with db:
        for front in fronts:
            db.insert_front(front)

    return fronts
