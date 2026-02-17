"""
Layer 3: Affiliation Impact Analysis

Tracks which organizations are publishing in each research front.
Identifies leading institutions per front and temporal trends.
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database


def _parse_json_field(value):
    """Parse JSON string or return as-is."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_affiliations(aff_str: str) -> List[str]:
    """
    Parse comma-separated affiliations string into list.

    Handles formats like:
    - "DeepMind, MIT, Tsinghua"
    - "Google DeepMind"
    - ""

    Returns:
        List of cleaned affiliation names
    """
    if not aff_str:
        return []

    affiliations = []
    for aff in aff_str.split(','):
        aff = aff.strip()
        if aff:
            # Normalize common variants
            aff = aff.replace('Google DeepMind', 'DeepMind')
            aff = aff.replace('Google Brain', 'Google')
            aff = aff.replace('OpenAI', 'OpenAI')
            affiliations.append(aff)

    return affiliations


def analyze_front_affiliations(fronts: List[dict],
                               db: Database) -> Dict[str, Dict]:
    """
    Analyze affiliation distribution across research fronts.

    Args:
        fronts: List of front dicts with core_papers
        db: Connected Database instance

    Returns:
        Dict mapping front_id -> {
            'affiliations': Counter of affiliation -> paper_count,
            'top_affiliations': List of (affiliation, count) tuples,
            'unique_affiliations': Set of affiliations unique to this front,
            'paper_count': Total papers in front
        }
    """
    front_affs = {}

    # First pass: collect all affiliations per front
    all_affiliations = set()
    for front in fronts:
        fid = front['front_id']
        papers = _parse_json_field(front.get('core_papers', []))

        aff_counter = Counter()
        for pid in papers:
            analysis = db.get_analysis(pid)
            if analysis:
                affs = _parse_affiliations(analysis.get('affiliations', ''))
                for aff in affs:
                    aff_counter[aff] += 1
                    all_affiliations.add(aff)

        front_affs[fid] = {
            'affiliations': aff_counter,
            'top_affiliations': aff_counter.most_common(5),
            'paper_count': len(papers),
        }

    # Second pass: identify unique affiliations per front
    for fid, info in front_affs.items():
        my_affs = set(info['affiliations'].keys())
        other_affs = set()
        for other_fid, other_info in front_affs.items():
            if other_fid != fid:
                other_affs.update(other_info['affiliations'].keys())

        info['unique_affiliations'] = sorted(my_affs - other_affs)
        info['shared_affiliations'] = sorted(my_affs & other_affs)

    return front_affs


def get_affiliation_portfolio(affiliations: List[str],
                              fronts: List[dict],
                              db: Database) -> Dict[str, List[str]]:
    """
    For given affiliations, show which fronts they're active in.

    Args:
        affiliations: List of affiliation names to track (e.g., ["DeepMind", "MIT"])
        fronts: List of front dicts
        db: Connected Database instance

    Returns:
        Dict mapping affiliation -> list of front_ids where they have papers
    """
    portfolio = defaultdict(list)

    for front in fronts:
        fid = front['front_id']
        papers = _parse_json_field(front.get('core_papers', []))

        front_affiliations = set()
        for pid in papers:
            analysis = db.get_analysis(pid)
            if analysis:
                affs = _parse_affiliations(analysis.get('affiliations', ''))
                front_affiliations.update(affs)

        for target_aff in affiliations:
            if target_aff in front_affiliations:
                portfolio[target_aff].append(fid)

    return dict(portfolio)


def get_top_affiliations_overall(db: Database,
                                 category: Optional[str] = None,
                                 min_papers: int = 2) -> List[Tuple[str, int]]:
    """
    Get most prolific affiliations across all papers.

    Args:
        db: Connected Database instance
        category: Filter by category (optional)
        min_papers: Minimum paper count to include

    Returns:
        List of (affiliation, paper_count) tuples, sorted by count desc
    """
    if category:
        papers = db.get_papers_by_category(category)
    else:
        papers = [dict(r) for r in db.fetchall(
            "SELECT affiliations FROM paper_analyses"
        )]

    aff_counter = Counter()
    for paper in papers:
        affs = _parse_affiliations(paper.get('affiliations', ''))
        for aff in affs:
            aff_counter[aff] += 1

    # Filter by min_papers
    top = [(aff, count) for aff, count in aff_counter.most_common()
           if count >= min_papers]

    return top


def format_affiliation_report(front_affs: Dict[str, Dict],
                              fronts: List[dict],
                              max_per_front: int = 3) -> str:
    """
    Format affiliation analysis as text report.

    Args:
        front_affs: Output from analyze_front_affiliations
        fronts: List of front dicts (for metadata)
        max_per_front: Max affiliations to show per front

    Returns:
        Formatted text string
    """
    lines = []
    lines.append("AFFILIATION IMPACT BY RESEARCH FRONT")
    lines.append("=" * 60)
    lines.append("")

    # Sort fronts by paper count descending
    fronts_sorted = sorted(fronts, key=lambda f: f.get('size', 0), reverse=True)

    for front in fronts_sorted:
        fid = front['front_id']
        if fid not in front_affs:
            continue

        info = front_affs[fid]
        size = front.get('size', 0)
        status = front.get('status', 'unknown')
        name = front.get('name', '')

        lines.append(f"Front: {fid}")
        if name and name != '[Enrichment failed]':
            lines.append(f"  Name: {name}")
        lines.append(f"  Size: {size} papers | Status: {status}")
        lines.append("")

        # Top affiliations
        top = info['top_affiliations'][:max_per_front]
        if top:
            lines.append(f"  Leading institutions ({len(top)}):")
            for aff, count in top:
                pct = (count / size * 100) if size > 0 else 0
                lines.append(f"    • {aff}: {count} papers ({pct:.0f}%)")

        # Unique affiliations
        unique = info.get('unique_affiliations', [])
        if unique:
            lines.append(f"  Unique to this front: {', '.join(unique[:3])}")

        lines.append("")

    return '\n'.join(lines)


def get_collaboration_pairs(fronts: List[dict],
                            db: Database,
                            min_coauthorships: int = 2) -> List[Tuple[str, str, int]]:
    """
    Identify affiliation pairs that frequently co-author.

    Args:
        fronts: List of front dicts
        db: Connected Database instance
        min_coauthorships: Minimum co-authored papers to include

    Returns:
        List of (affiliation1, affiliation2, coauthor_count) tuples
    """
    from itertools import combinations

    pair_counter = Counter()

    for front in fronts:
        papers = _parse_json_field(front.get('core_papers', []))

        for pid in papers:
            analysis = db.get_analysis(pid)
            if not analysis:
                continue

            affs = _parse_affiliations(analysis.get('affiliations', ''))
            # Count unique pairs (sorted to avoid duplicates)
            for aff1, aff2 in combinations(sorted(set(affs)), 2):
                pair_counter[(aff1, aff2)] += 1

    # Filter and sort
    pairs = [(a1, a2, count) for (a1, a2), count in pair_counter.items()
             if count >= min_coauthorships]
    pairs.sort(key=lambda x: x[2], reverse=True)

    return pairs
