"""
Layer 3: Framework Genealogy Tracker

Tracks framework_lineage field over time to visualize research lineages.
Identifies parent frameworks, variants, and temporal progression.
"""

import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database


def _parse_json_field(value):
    """Parse JSON string or return as-is."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD date string."""
    try:
        return datetime.strptime(date_str[:10], '%Y-%m-%d')
    except (ValueError, TypeError):
        return datetime.now()


def extract_framework_lineages(db: Database,
                               category: Optional[str] = None,
                               min_date: Optional[str] = None) -> Dict[str, List[dict]]:
    """
    Extract all papers with framework_lineage tags, grouped by framework.

    Args:
        db: Connected Database instance
        category: Filter by category (optional)
        min_date: Only include papers published on or after this date (YYYY-MM-DD)

    Returns:
        Dict mapping framework_name -> list of paper dicts with that lineage
    """
    if category:
        papers = db.get_papers_by_category(category)
    else:
        papers = [dict(r) for r in db.fetchall(
            "SELECT * FROM paper_analyses ORDER BY published_date DESC"
        )]

    lineages = defaultdict(list)

    for paper in papers:
        pub_date = paper.get('published_date', '')
        if min_date and pub_date < min_date:
            continue

        tags = _parse_json_field(paper.get('tags'))
        framework = tags.get('framework_lineage')

        if framework:
            lineages[framework].append({
                'arxiv_id': paper.get('arxiv_id'),
                'title': paper.get('title', ''),
                'published_date': pub_date,
                'affiliations': paper.get('affiliations', ''),
                'must_read': _parse_json_field(paper.get('significance', {})).get('must_read', False),
            })

    # Sort each framework's papers by date
    for framework in lineages:
        lineages[framework].sort(key=lambda p: p['published_date'])

    return dict(lineages)


def build_genealogy_tree(lineages: Dict[str, List[dict]]) -> Dict[str, Dict]:
    """
    Build a genealogy tree showing framework parent-child relationships.

    Uses heuristics to infer parent frameworks:
    - Substring matching (e.g., "alphaevolve_sage" → parent "alphaevolve")
    - Temporal ordering (earlier frameworks are potential parents)

    Args:
        lineages: Dict from extract_framework_lineages

    Returns:
        Dict mapping framework_name -> {
            'parent': parent_framework_name or None,
            'children': list of child framework names,
            'first_paper_date': earliest paper date,
            'paper_count': number of papers
        }
    """
    tree = {}
    frameworks = sorted(lineages.keys())

    # Initialize tree nodes
    for fw in frameworks:
        papers = lineages[fw]
        tree[fw] = {
            'parent': None,
            'children': [],
            'first_paper_date': papers[0]['published_date'] if papers else '',
            'paper_count': len(papers),
            'latest_paper_date': papers[-1]['published_date'] if papers else '',
        }

    # Infer parent relationships using substring matching
    for fw in frameworks:
        # Check if this framework name contains another framework as substring
        # Example: "alphaevolve_sage" contains "alphaevolve"
        for potential_parent in frameworks:
            if potential_parent != fw and potential_parent in fw:
                # Check temporal order (parent should be earlier)
                if tree[potential_parent]['first_paper_date'] <= tree[fw]['first_paper_date']:
                    tree[fw]['parent'] = potential_parent
                    tree[potential_parent]['children'].append(fw)
                    break

    return tree


def format_genealogy_text(lineages: Dict[str, List[dict]],
                          tree: Dict[str, Dict],
                          show_papers: bool = False,
                          max_depth: int = 3) -> str:
    """
    Format genealogy tree as indented text.

    Args:
        lineages: Framework lineages from extract_framework_lineages
        tree: Genealogy tree from build_genealogy_tree
        show_papers: Include paper titles under each framework
        max_depth: Maximum tree depth to display

    Returns:
        Formatted text string
    """
    lines = []

    # Find root frameworks (no parent)
    roots = [fw for fw, info in tree.items() if info['parent'] is None]
    roots.sort(key=lambda fw: tree[fw]['first_paper_date'])

    def _format_node(fw: str, depth: int = 0, prefix: str = ''):
        if depth > max_depth:
            return

        info = tree[fw]
        papers = lineages[fw]

        # Framework header
        indent = '  ' * depth
        connector = '└─> ' if depth > 0 else ''
        date_range = f"{info['first_paper_date'][:4]}"
        if info['latest_paper_date'][:4] != info['first_paper_date'][:4]:
            date_range += f"-{info['latest_paper_date'][:4]}"

        must_read_count = sum(1 for p in papers if p['must_read'])
        badge = f" [{must_read_count} must-read]" if must_read_count > 0 else ""

        lines.append(f"{indent}{connector}{fw} ({date_range}) — {info['paper_count']} papers{badge}")

        # Show papers if requested
        if show_papers and depth < 2:  # Only show papers for top 2 levels
            for paper in papers[:3]:  # Show max 3 papers per framework
                paper_indent = '  ' * (depth + 1)
                title_short = paper['title'][:60] + '...' if len(paper['title']) > 60 else paper['title']
                must_read_marker = ' ⭐' if paper['must_read'] else ''
                lines.append(f"{paper_indent}• [{paper['published_date'][:7]}] {title_short}{must_read_marker}")
            if len(papers) > 3:
                lines.append(f"{paper_indent}  ... and {len(papers) - 3} more")

        # Recurse to children
        children = sorted(info['children'], key=lambda c: tree[c]['first_paper_date'])
        for child in children:
            _format_node(child, depth + 1, prefix + '  ')

    for root in roots:
        _format_node(root)
        lines.append('')  # Blank line between root trees

    return '\n'.join(lines)


def get_active_frameworks(lineages: Dict[str, List[dict]],
                         days: int = 30) -> List[Tuple[str, int]]:
    """
    Get frameworks with papers published in the last N days.

    Args:
        lineages: Framework lineages from extract_framework_lineages
        days: Lookback window in days

    Returns:
        List of (framework_name, recent_paper_count) tuples, sorted by count desc
    """
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    active = []
    for fw, papers in lineages.items():
        recent = [p for p in papers if p['published_date'] >= cutoff]
        if recent:
            active.append((fw, len(recent)))

    active.sort(key=lambda x: x[1], reverse=True)
    return active


def get_framework_summary(lineages: Dict[str, List[dict]],
                         tree: Dict[str, Dict]) -> Dict:
    """
    Get summary statistics about framework landscape.

    Returns:
        Dict with keys: total_frameworks, root_frameworks, active_last_30d,
                       top_frameworks, newest_frameworks
    """
    roots = [fw for fw, info in tree.items() if info['parent'] is None]
    active = get_active_frameworks(lineages, days=30)

    # Top frameworks by paper count
    top = sorted(lineages.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    # Newest frameworks (first paper in last 90 days)
    from datetime import datetime, timedelta
    cutoff_90d = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    newest = [(fw, info) for fw, info in tree.items()
              if info['first_paper_date'] >= cutoff_90d]
    newest.sort(key=lambda x: x[1]['first_paper_date'], reverse=True)

    return {
        'total_frameworks': len(lineages),
        'root_frameworks': len(roots),
        'active_last_30d': len(active),
        'top_frameworks': [(fw, len(papers)) for fw, papers in top],
        'newest_frameworks': [(fw, info['first_paper_date']) for fw, info in newest[:5]],
        'roots': roots,
    }
