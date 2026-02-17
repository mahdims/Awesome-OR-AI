"""
Layer 3: Graph Visualizer

Generates PNG visualizations of co-citation/front networks per category.
Nodes colored by front membership, sized by relevance, edges by co-citation strength.
"""

import json
import io
import base64
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from db.database import Database

# Color palette for fronts (up to 10 fronts)
FRONT_COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
    '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
]

STATUS_MARKERS = {
    'emerging': '^',    # triangle up
    'growing': 'o',     # circle
    'stable': 's',      # square
    'declining': 'v',   # triangle down
    'merged': 'D',      # diamond
    'split': 'p',       # pentagon
}


def _get_cocitation_graph(db: Database, category: str,
                          snapshot_date: str) -> nx.Graph:
    """Load co-citation edges from DB into a NetworkX graph."""
    rows = db.fetchall(
        """SELECT paper1_id, paper2_id, strength
           FROM cocitation_edges
           WHERE category = ? AND snapshot_date = ?""",
        (category, snapshot_date)
    )
    G = nx.Graph()
    for r in rows:
        G.add_edge(r['paper1_id'], r['paper2_id'],
                    weight=r['strength'] or 0.5)
    return G


def _build_front_map(fronts: List[dict]) -> Dict[str, int]:
    """Map paper_id -> front_index."""
    paper_front = {}
    for i, front in enumerate(fronts):
        papers = front.get('core_papers', [])
        if isinstance(papers, str):
            papers = json.loads(papers)
        for pid in papers:
            paper_front[pid] = i
    return paper_front


def visualize_fronts(category: str, db: Database,
                     output_path: Optional[Path] = None,
                     figsize: tuple = (12, 8)) -> Optional[str]:
    """
    Generate a front network visualization for a category.

    Args:
        category: Category name
        db: Connected Database instance
        output_path: Path to save PNG. If None, returns base64 string.
        figsize: Figure size in inches

    Returns:
        Base64-encoded PNG string if output_path is None, else None (saves file).
    """
    fronts = db.get_latest_fronts(category)
    if not fronts:
        return None

    snapshot_date = fronts[0].get('snapshot_date', date.today().isoformat())
    cocitation = _get_cocitation_graph(db, category, snapshot_date)

    if cocitation.number_of_nodes() == 0:
        return None

    paper_front = _build_front_map(fronts)

    # Assign colors and sizes
    node_colors = []
    node_sizes = []
    for node in cocitation.nodes():
        front_idx = paper_front.get(node, -1)
        if front_idx >= 0:
            node_colors.append(FRONT_COLORS[front_idx % len(FRONT_COLORS)])
        else:
            node_colors.append('#CCCCCC')

        # Size by relevance score (fetch from DB)
        analysis = db.get_analysis(node)
        if analysis:
            rel = analysis.get('relevance', '{}')
            if isinstance(rel, str):
                try:
                    rel = json.loads(rel)
                except json.JSONDecodeError:
                    rel = {}
            max_score = max(
                rel.get('methodological', 0),
                rel.get('problem', 0),
                rel.get('inspirational', 0),
            )
            node_sizes.append(100 + max_score * 50)
        else:
            node_sizes.append(100)

    # Edge widths by weight
    edge_weights = [cocitation[u][v].get('weight', 0.5) * 2
                    for u, v in cocitation.edges()]

    # Layout
    pos = nx.spring_layout(cocitation, k=1.5, iterations=50, seed=42)

    # Draw
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor('#FAFAFA')

    nx.draw_networkx_edges(cocitation, pos, ax=ax,
                           width=edge_weights, alpha=0.3,
                           edge_color='#888888')

    nx.draw_networkx_nodes(cocitation, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.85, edgecolors='white', linewidths=0.5)

    # Short labels (last 4 chars of arxiv_id)
    labels = {n: n[-4:] if len(n) > 4 else n for n in cocitation.nodes()}
    nx.draw_networkx_labels(cocitation, pos, labels, ax=ax,
                            font_size=6, font_color='#333333')

    # Legend
    legend_patches = []
    for i, front in enumerate(fronts):
        fid = front.get('front_id', f'Front {i}')
        short_id = fid.split('_front_')[-1] if '_front_' in fid else str(i)
        status = front.get('status', '?')
        size = front.get('size', '?')
        color = FRONT_COLORS[i % len(FRONT_COLORS)]
        label = f"Front {short_id} ({size} papers, {status})"
        legend_patches.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=legend_patches, loc='upper left',
              fontsize=7, framealpha=0.9)

    # Clean slug for title
    slug = category.lower().replace(' ', '_')[:30]
    ax.set_title(f"Research Fronts: {category}\n{snapshot_date}",
                 fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('ascii')


def visualize_all_categories(db: Database,
                             output_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Generate visualizations for all categories.

    Returns dict mapping category -> base64 PNG string (or file path if output_dir set).
    """
    from layer3.data_collector import get_categories

    results = {}
    categories = get_categories(db)

    for category in categories:
        slug = category.lower().replace(' ', '_').replace('/', '_')[:30]

        if output_dir:
            out_path = output_dir / f"{slug}_{date.today().isoformat()}.png"
            visualize_fronts(category, db, output_path=out_path)
            results[category] = str(out_path)
        else:
            b64 = visualize_fronts(category, db)
            if b64:
                results[category] = b64

    return results
