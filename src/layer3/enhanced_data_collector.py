"""
Layer 3: Enhanced Data Collector

Extends base data_collector with smart prioritization, framework genealogy,
and affiliation impact analysis.
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database
from layer3.data_collector import collect_weekly_data, _enrich_paper
from layer3.smart_prioritization import rank_papers, get_priority_reasons
from layer3.framework_genealogy import (
    extract_framework_lineages,
    build_genealogy_tree,
    format_genealogy_text,
    get_active_frameworks,
    get_framework_summary,
)
from layer3.affiliation_analysis import (
    analyze_front_affiliations,
    get_top_affiliations_overall,
    format_affiliation_report,
    get_collaboration_pairs,
)


def collect_enhanced_weekly_data(category: str, db: Database,
                                 days: int = 7) -> dict:
    """
    Collect data for enhanced weekly email with:
    - Smart paper prioritization
    - Framework genealogy
    - Affiliation impact analysis
    - Standard weekly data (papers, fronts, bridges)

    Returns dict with all weekly data plus:
        - priority_papers: top papers ranked by composite score
        - framework_genealogy: lineages, tree, summary
        - affiliation_analysis: per-front and overall stats
    """
    # Get base weekly data
    data = collect_weekly_data(category, db, days=days)

    papers = data['papers']
    fronts = data['fronts']
    bridges = data['bridges']

    # --- 1. Smart Paper Prioritization ---
    # Build front_map: paper_id -> front_status
    front_map = {}
    for front in fronts:
        status = front.get('status', 'unknown')
        for pid in front.get('core_papers', []):
            front_map[pid] = status

    # Build bridge_map: paper_id -> bridge_score
    bridge_map = {}
    for bridge in bridges:
        pid = bridge.get('paper_id')
        score = bridge.get('bridge_score', 0)
        if pid:
            bridge_map[pid] = score

    # Rank papers
    priority_papers = rank_papers(
        papers,
        front_map=front_map,
        bridge_map=bridge_map,
        days_window=days,
        top_n=10
    )

    # Add priority reasons
    for item in priority_papers:
        paper = item['paper']
        pid = paper.get('arxiv_id', '')
        item['reasons'] = get_priority_reasons(
            paper,
            front_status=front_map.get(pid),
            bridge_score=bridge_map.get(pid, 0.0)
        )

    data['priority_papers'] = priority_papers

    # --- 2. Framework Genealogy ---
    lineages = extract_framework_lineages(db, category=category)
    tree = build_genealogy_tree(lineages)
    genealogy_text = format_genealogy_text(lineages, tree, show_papers=False)
    active_frameworks = get_active_frameworks(lineages, days=days)
    framework_summary = get_framework_summary(lineages, tree)

    data['framework_genealogy'] = {
        'lineages': lineages,
        'tree': tree,
        'text': genealogy_text,
        'active_last_30d': active_frameworks,
        'summary': framework_summary,
    }

    # --- 3. Affiliation Analysis ---
    front_affiliations = analyze_front_affiliations(fronts, db)
    top_affiliations = get_top_affiliations_overall(db, category=category, min_papers=2)
    affiliation_report = format_affiliation_report(front_affiliations, fronts)
    collaboration_pairs = get_collaboration_pairs(fronts, db, min_coauthorships=2)

    data['affiliation_analysis'] = {
        'per_front': front_affiliations,
        'top_overall': top_affiliations[:10],
        'report_text': affiliation_report,
        'collaborations': collaboration_pairs[:5],
    }

    # --- 4. Enhanced Stats ---
    data['stats'].update({
        'top_priority_count': len(priority_papers),
        'active_frameworks_count': len(active_frameworks),
        'top_affiliations_count': len(top_affiliations[:10]),
    })

    return data
