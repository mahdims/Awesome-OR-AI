"""
Layer 2: Bridge Paper Detection

Identifies papers that connect multiple research fronts within a category.
Bridge papers have citations/co-citations spanning different communities.
"""

import json
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from db.database import Database


def detect_bridge_papers(cocitation_graph: nx.Graph,
                          fronts: List[Dict],
                          category: str,
                          snapshot_date: Optional[str] = None,
                          min_bridge_score: float = 0.5) -> List[Dict]:
    """
    Identify papers that connect multiple research fronts.

    A bridge paper belongs to one front (home front) but has co-citation
    edges to papers in other fronts. The bridge score measures what
    fraction of its edges cross front boundaries.

    Args:
        cocitation_graph: Undirected weighted co-citation graph
        fronts: List of front dicts (with 'front_id' and 'core_papers')
        category: Category name
        snapshot_date: Date string (YYYY-MM-DD)
        min_bridge_score: Minimum score to qualify as bridge paper

    Returns:
        List of bridge paper dicts ready for database insertion
    """
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    # Build paper -> front mapping
    paper_to_front = {}
    for front in fronts:
        for paper_id in front['core_papers']:
            paper_to_front[paper_id] = front['front_id']

    bridges = []

    for paper_id, home_front_id in paper_to_front.items():
        if paper_id not in cocitation_graph:
            continue

        neighbors = list(cocitation_graph.neighbors(paper_id))
        if not neighbors:
            continue

        # Accumulate edge weight per front (use 'strength' attribute if present,
        # fallback to 'weight', then 1.0 for unweighted graphs).
        front_edge_weights: dict = defaultdict(float)
        total_weight = 0.0

        for neighbor in neighbors:
            neighbor_front = paper_to_front.get(neighbor)
            if neighbor_front:
                edge_data = cocitation_graph.get_edge_data(paper_id, neighbor) or {}
                w = edge_data.get('strength', edge_data.get('weight', 1.0))
                front_edge_weights[neighbor_front] += w
                total_weight += w

        if total_weight == 0.0:
            continue

        # Bridge score: fraction of total edge weight crossing front boundaries
        home_weight = front_edge_weights.get(home_front_id, 0.0)
        cross_weight = total_weight - home_weight
        bridge_score = cross_weight / total_weight

        if bridge_score < min_bridge_score:
            continue

        # Identify which other fronts this paper connects to
        connected_fronts = [
            fid for fid in front_edge_weights
            if fid != home_front_id and front_edge_weights[fid] > 0
        ]

        if not connected_fronts:
            continue

        bridges.append({
            'paper_id': paper_id,
            'category': category,
            'snapshot_date': snapshot_date,
            'home_front_id': home_front_id,
            'connected_fronts': connected_fronts,
            'bridge_score': round(bridge_score, 4),
        })

    # Sort by bridge score descending
    bridges.sort(key=lambda x: x['bridge_score'], reverse=True)

    print(f"  Detected {len(bridges)} bridge papers (min_score={min_bridge_score})")
    for b in bridges[:5]:
        print(f"    {b['paper_id']}: score={b['bridge_score']:.3f}, "
              f"connects {len(b['connected_fronts'])} fronts")

    return bridges


def store_bridge_papers(bridges: List[Dict], db: Database):
    """Store bridge papers in the database."""
    if bridges:
        db.insert_bridge_papers(bridges)
        print(f"  Stored {len(bridges)} bridge papers in database")
