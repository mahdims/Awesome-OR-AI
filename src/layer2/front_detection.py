"""
Layer 2: Co-citation Network & Research Front Detection

Builds co-citation networks from citation graphs and applies
Louvain community detection to identify research fronts.
"""

import json
from collections import Counter, defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

try:
    import community as community_louvain  # python-louvain
except ImportError:
    community_louvain = None
    print("[WARN] python-louvain not installed. Install with: pip install python-louvain")

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
            ancestors, framework_lineage, specific_domain, llm_coupling)


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

    print(f"  Semantic edges: {len(analyses)}/{len(corpus_list)} papers have Layer 1 data "
          f"({missing} skipped)")
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
              _anc, fw_lineage, sp_domain, llm_cpl) in analyses.items():
        for t in methods | problems:
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

    # Anchor scale to mean existing citation weight
    mean_w = (sum(d['weight'] for _, _, d in G.edges(data=True)) / G.number_of_edges()
              if G.number_of_edges() > 0 else 1.0)

    ids = sorted(analyses.keys())
    tag_new = lineage_new = 0

    # Signals 1–8: IDF-weighted pairwise similarity
    for i in range(len(ids)):
        pid_a = ids[i]
        m_a, p_a, cls_a, role_a, stype_a, bench_a, _p, _anc_a, lin_a, dom_a, cpl_a = analyses[pid_a]
        tags_a = m_a | p_a

        for j in range(i + 1, len(ids)):
            pid_b = ids[j]
            m_b, p_b, cls_b, role_b, stype_b, bench_b, _p, _anc_b, lin_b, dom_b, cpl_b = analyses[pid_b]
            tags_b = m_b | p_b

            # Signal 1: IDF-weighted tag Jaccard (methods + problems)
            idf_tag = idf_jaccard(tags_a, tags_b, tag_df)

            # Signal 2: IDF-scaled class homophily
            # Near-zero when all papers share the same class (pure noise)
            class_bonus = (idf(class_df[cls_a]) * 0.30
                           if (cls_a and cls_a == cls_b) else 0.0)

            # Signal 3: IDF-scaled LLM role match
            role_bonus = (idf(role_df[role_a]) * 0.25
                          if (role_a and role_a == role_b) else 0.0)

            # Signal 4: IDF-scaled search type match
            stype_bonus = (idf(stype_df[stype_a]) * 0.15
                           if (stype_a and stype_a == stype_b) else 0.0)

            # Signal 5: IDF-weighted benchmark Jaccard (benchmark names are specific)
            idf_bench = idf_jaccard(bench_a, bench_b, bench_df)

            # Signal 6: IDF-scaled framework lineage match (e.g. both "alphaevolve")
            # Zero when all papers share the same lineage — IDF noise-proofs this
            lineage_bonus = (idf(lineage_df[lin_a]) * 0.40
                             if (lin_a and lin_a == lin_b) else 0.0)

            # Signal 7: IDF-scaled specific domain match (e.g. "combinatorial_routing")
            domain_bonus = (idf(domain_df[dom_a]) * 0.40
                            if (dom_a and dom_a == dom_b) else 0.0)

            # Signal 8: IDF-scaled LLM coupling match (e.g. "rl_trained")
            coupling_bonus = (idf(coupling_df[cpl_a]) * 0.20
                              if (cpl_a and cpl_a == cpl_b) else 0.0)

            raw = (idf_tag + class_bonus + role_bonus + stype_bonus + idf_bench
                   + lineage_bonus + domain_bonus + coupling_bonus)

            # Threshold: skip weak / noise connections
            if raw < min_similarity:
                continue

            w = raw * mean_w * semantic_weight
            if G.has_edge(pid_a, pid_b):
                G[pid_a][pid_b]['weight'] += w
            else:
                G.add_edge(pid_a, pid_b, weight=w)
                tag_new += 1

    # Signal 9: Lineage edges (only exact arxiv_id matches in corpus)
    # ancestors is index 7 in the 11-field tuple
    for pid_a, (_m, _p, _cls, _role, _stype, _bench, _props,
                ancestors, _lin, _dom, _cpl) in analyses.items():
        for entry in ancestors:
            anc = entry.get('paper', '')
            if anc not in corpus_ids:
                continue  # title-only refs or out-of-corpus papers: skip silently
            w = mean_w * semantic_weight
            if G.has_edge(pid_a, anc):
                G[pid_a][anc]['weight'] += w
            else:
                G.add_edge(pid_a, anc, weight=w)
                lineage_new += 1

    print(f"  Semantic edges: {tag_new} pairwise (threshold={min_similarity}), "
          f"{lineage_new} lineage edges added")


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
    if community_louvain is None:
        raise ImportError("python-louvain required: pip install python-louvain")

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

    # Run Louvain community detection
    partition = community_louvain.best_partition(
        cocitation_graph,
        weight='weight',
        resolution=resolution,
        random_state=42
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
    print(f"\n  Running Louvain community detection...")
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
