"""
Layer 2: Citation Graph Builder

Fetches citation data from Semantic Scholar API and builds
directed citation graphs per category using NetworkX.
"""

import time
import requests
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests
RATE_LIMIT_WAIT = 5  # seconds on 429
MAX_RETRIES = 3


class CitationGraphBuilder:
    """
    Builds directed citation graphs per category.

    For each analyzed paper in a category, fetches its references
    from Semantic Scholar. The resulting graph has:
    - Nodes: paper IDs (arxiv_id or S2 paper ID)
    - Edges: paper A -> paper B means A cites B
    """

    def __init__(self, s2_api_key: Optional[str] = None):
        self.db = Database()
        self.session = requests.Session()
        if s2_api_key:
            self.session.headers['x-api-key'] = s2_api_key
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _s2_get(self, url: str, params: dict = None, timeout: int = 30) -> Optional[dict]:
        """Make a rate-limited Semantic Scholar API call with retries."""
        for attempt in range(MAX_RETRIES):
            self._rate_limit()
            try:
                r = self.session.get(url, params=params, timeout=timeout)
                if r.status_code == 429:
                    wait = RATE_LIMIT_WAIT * (attempt + 1)
                    print(f"    [S2] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.json()
            except requests.exceptions.Timeout:
                print(f"    [S2] Timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            except requests.exceptions.RequestException as e:
                print(f"    [S2] Error: {e} (attempt {attempt + 1}/{MAX_RETRIES})")
        return None

    def fetch_references(self, arxiv_id: str) -> Optional[List[str]]:
        """
        Fetch papers referenced by the given paper.

        Returns list of arxiv IDs that this paper cites, or None on API failure
        (None is distinct from [] which means the paper has no references).
        """
        url = f"{SEMANTIC_SCHOLAR_URL}ArXiv:{arxiv_id}/references"
        params = {"fields": "externalIds", "limit": 500}

        result = self._s2_get(url, params)
        if result is None:
            return None  # API failure — do not log as fetched

        refs = []
        for item in result.get("data", []):
            cited = item.get("citedPaper", {})
            ext_ids = cited.get("externalIds") or {}
            ref_arxiv = ext_ids.get("ArXiv")
            if ref_arxiv:
                refs.append(ref_arxiv)

        return refs

    def fetch_citations(self, arxiv_id: str) -> Optional[List[str]]:
        """
        Fetch papers that cite the given paper.

        Returns list of arxiv IDs that cite this paper, or None if the first
        API page fails (None is distinct from [] which means no citations).
        Partial results from later page failures are returned as-is.
        """
        url = f"{SEMANTIC_SCHOLAR_URL}ArXiv:{arxiv_id}/citations"
        params = {"fields": "externalIds", "limit": 500}

        all_citing = []
        offset = 0
        first_page = True

        while True:
            params["offset"] = offset
            result = self._s2_get(url, params)
            if result is None:
                if first_page:
                    return None  # First page failed — treat as API failure
                break  # Later page failed — return partial results

            first_page = False
            data = result.get("data", [])
            if not data:
                break

            for item in data:
                citing = item.get("citingPaper", {})
                ext_ids = citing.get("externalIds") or {}
                citing_arxiv = ext_ids.get("ArXiv")
                if citing_arxiv:
                    all_citing.append(citing_arxiv)

            # Check for more pages
            if result.get("next"):
                offset = result["next"]
            else:
                break

        return all_citing

    def build_citation_graph(self, category: str,
                              fetch_mode: str = "references",
                              force_refresh: bool = False) -> nx.DiGraph:
        """
        Build (or update) the citation graph for a category.

        Incremental by default — only corpus papers not yet in
        citation_fetch_log are queried against Semantic Scholar.
        Previously fetched papers reuse their edges from the citations
        table, so adding one new paper costs one API call, not N.

        When a new paper joins the corpus its outgoing references
        (new → old/external) and incoming citations (old/external → new)
        are fetched and stored.  The full graph is then reconstructed
        from the citations table, which already contains the cross-paper
        edges between new and existing corpus papers.

        Args:
            category:       Category name
            fetch_mode:     "references" — outgoing edges only
                            "both"       — outgoing + incoming edges
            force_refresh:  If True, re-fetch ALL corpus papers and
                            reset the fetch log (ignores cache).

        Returns:
            Directed NetworkX graph loaded entirely from the DB.
        """
        print(f"\n{'='*60}")
        print(f"Building citation graph: {category}")
        print(f"Mode: {fetch_mode}  |  Force refresh: {force_refresh}")
        print(f"{'='*60}")

        with self.db as db:
            papers = db.get_papers_by_category(category)

        paper_ids = [p['arxiv_id'] for p in papers]
        corpus_set = set(paper_ids)
        print(f"Corpus papers in category: {len(corpus_set)}")

        if not corpus_set:
            print("[WARN] No analyzed papers found for this category")
            return nx.DiGraph()

        # Determine which papers need fetching
        if force_refresh:
            with self.db as db:
                # Clear both the fetch log AND all citation edges so stale
                # rows from a previous fetch are fully removed.
                db.clear_fetch_log(category)
                db.clear_citations(category)
            to_fetch = paper_ids
            print(f"  Force refresh: will re-fetch all {len(to_fetch)} papers")
        else:
            with self.db as db:
                to_fetch = db.get_unfetched_papers(paper_ids, category, fetch_mode)
            cached = len(corpus_set) - len(to_fetch)
            print(f"  Cached: {cached} papers  |  To fetch: {len(to_fetch)} papers")

        # Fetch only the papers that are missing from the log.
        # Edges are inserted immediately after each paper so that a crash
        # mid-loop leaves the DB consistent with the fetch log.
        if to_fetch:
            total_edges = 0

            for i, arxiv_id in enumerate(to_fetch, 1):
                print(f"  [{i}/{len(to_fetch)}] Fetching references for {arxiv_id}...", end=" ")
                refs = self.fetch_references(arxiv_id)
                if refs is None:
                    print("API failure — skipping (not logged)")
                    continue
                print(f"{len(refs)} references found")

                paper_edges = [(arxiv_id, ref_id) for ref_id in refs]

                cited_count = 0
                if fetch_mode == "both":
                    print(f"           Fetching citations...", end=" ")
                    citing = self.fetch_citations(arxiv_id)
                    if citing is None:
                        print("API failure — skipping (not logged)")
                        continue
                    print(f"{len(citing)} citations found")
                    paper_edges.extend([(cit_id, arxiv_id) for cit_id in citing])
                    cited_count = len(citing)

                # Persist this paper's edges and log in one atomic block so a
                # crash can never leave the fetch log ahead of the stored edges.
                with self.db as db:
                    db.insert_citations(paper_edges, category)
                    db.log_citation_fetch(
                        arxiv_id, category, fetch_mode,
                        refs_count=len(refs),
                        cited_count=cited_count
                    )

                total_edges += len(paper_edges)

            print(f"\n  Stored {total_edges} new citation edges")
        else:
            print("  All papers cached — loading graph from database")

        # Always reconstruct the full graph from the DB
        G = self.load_citation_graph_from_db(category)

        print(f"\nGraph built:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Corpus nodes: {sum(1 for _, d in G.nodes(data=True) if d.get('in_corpus'))}")

        return G

    def load_citation_graph_from_db(self, category: str) -> nx.DiGraph:
        """Load previously built citation graph from database."""
        with self.db as db:
            rows = db.fetchall(
                "SELECT source_paper_id, target_paper_id FROM citations WHERE category = ?",
                (category,)
            )
            papers = db.get_papers_by_category(category)

        paper_ids = {p['arxiv_id'] for p in papers}

        G = nx.DiGraph()
        for p in papers:
            G.add_node(p['arxiv_id'], title=p.get('title', ''),
                       in_corpus=True)

        for row in rows:
            src, tgt = row['source_paper_id'], row['target_paper_id']
            if src not in G:
                G.add_node(src, in_corpus=(src in paper_ids))
            if tgt not in G:
                G.add_node(tgt, in_corpus=(tgt in paper_ids))
            G.add_edge(src, tgt)

        return G

    def get_corpus_subgraph(self, G: nx.DiGraph) -> nx.DiGraph:
        """Extract subgraph containing only corpus (analyzed) papers."""
        corpus_nodes = [n for n, d in G.nodes(data=True) if d.get('in_corpus')]
        return G.subgraph(corpus_nodes).copy()
