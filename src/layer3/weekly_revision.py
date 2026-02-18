"""
Layer 3: Weekly Living Review Revision

Restructures the living review markdown docs using Layer 2 front analysis.
Reorganizes papers by research fronts, updates front summaries,
and adds bridge paper cross-references.
"""

import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database
from layer3.data_collector import collect_weekly_data, _json
from layer3.daily_update import _category_slug, _paper_to_markdown, LIVING_REVIEWS_DIR


def _front_short_id(front_id: str) -> str:
    if '_front_' in front_id:
        return front_id.split('_front_')[-1]
    return front_id[-4:]


def _front_section(front: dict, front_methods: dict,
                   all_front_methods: dict) -> str:
    """Generate markdown for a single research front."""
    fid = front['front_id']
    short_id = _front_short_id(fid)
    status = front.get('status', 'unknown')
    size = front.get('size', 0)
    density = front.get('internal_density', 0)
    summary = front.get('summary', '')
    dom_methods = front.get('dominant_methods', [])
    dom_problems = front.get('dominant_problems', [])

    lines = [
        f"### Front {short_id} ({size} papers) — {status.upper()}",
        f"",
        f"**Density:** {density:.2f} | **Methods:** {', '.join(dom_methods[:5])} | **Problems:** {', '.join(dom_problems[:5])}",
        f"",
    ]

    # Unique vs shared methods
    my_methods = set(front_methods.get(fid, []))
    other_methods = set()
    for other_fid, other_m in all_front_methods.items():
        if other_fid != fid:
            other_methods.update(other_m)
    unique = sorted(my_methods - other_methods)
    shared = sorted(my_methods & other_methods)
    if unique:
        lines.append(f"*Unique methods:* {', '.join(unique)}")
    if shared:
        lines.append(f"*Shared methods:* {', '.join(shared)}")
    if unique or shared:
        lines.append("")

    # Summary
    if summary:
        lines.append(summary)
        lines.append("")

    # Papers
    lines.append("**Papers:**")
    lines.append("")
    for paper in front.get('papers_detail', []):
        lines.append(_paper_to_markdown(paper))

    return '\n'.join(lines)


def _bridge_section(bridges: List[dict]) -> str:
    """Generate markdown for bridge papers."""
    if not bridges:
        return ""

    lines = ["## Bridge Papers", "", "Papers connecting multiple research fronts:", ""]

    for bridge in bridges:
        pid = bridge.get('paper_id', '')
        score = bridge.get('bridge_score', 0)
        verdict = bridge.get('verdict', '?')
        home = _front_short_id(bridge.get('home_front_id', ''))
        connected = [_front_short_id(c) for c in bridge.get('connected_fronts', [])]

        paper = bridge.get('paper')
        title = paper.get('title', pid) if paper else pid
        brief = (paper.get('brief', '') if paper else '')[:200]

        lines.append(f"### [{title}](https://arxiv.org/abs/{pid})")
        lines.append(f"")
        lines.append(f"**{verdict}** | score={score:.2f} | Front {home} → {', '.join('Front ' + c for c in connected)}")
        lines.append(f"")
        if brief:
            lines.append(f"> {brief}")
            lines.append(f"")

    return '\n'.join(lines)


def revise_living_review(category: str, db: Database,
                         days: int = 7) -> bool:
    """
    Restructure the living review markdown using Layer 2 front analysis.

    Replaces the "Research Fronts" and "Bridge Papers" sections entirely.
    Keeps the "Recent Papers" section intact.

    Args:
        category: Category name
        db: Connected Database instance
        days: Lookback window for weekly data

    Returns:
        True if revision was performed.
    """
    LIVING_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    slug = _category_slug(category)
    review_path = LIVING_REVIEWS_DIR / f"{slug}.md"

    if not review_path.exists():
        print(f"  [SKIP] No living review found for {category}. Run daily update first.")
        return False

    # Collect data
    data = collect_weekly_data(category, db, days=days)
    fronts = data.get('fronts', [])
    bridges = data.get('bridges', [])
    front_methods = data.get('front_methods', {})

    if not fronts:
        print(f"  [SKIP] No fronts detected for {category}.")
        return False

    content = review_path.read_text(encoding='utf-8')

    # Build new Research Fronts section
    fronts_md = ["## Research Fronts", "",
                 f"*{len(fronts)} fronts detected — snapshot {fronts[0].get('snapshot_date', '?')}*", ""]
    for front in fronts:
        fronts_md.append(_front_section(front, front_methods, front_methods))
        fronts_md.append("")
    fronts_text = '\n'.join(fronts_md)

    # Build new Bridge Papers section
    bridges_text = _bridge_section(bridges)

    # Replace sections in the document
    # Replace "## Research Fronts" ... "## Bridge Papers" or "---"
    fronts_pattern = r'## Research Fronts.*?(?=## Bridge Papers|---\s*\n\*Generated)'
    if re.search(fronts_pattern, content, re.DOTALL):
        _ft = fronts_text + '\n\n'
        content = re.sub(fronts_pattern, lambda m: _ft, content, flags=re.DOTALL)
    else:
        # Append if marker not found
        content += '\n\n' + fronts_text

    # Replace bridge papers section
    bridge_pattern = r'## Bridge Papers.*?(?=---\s*\n\*Generated|\Z)'
    if bridges_text:
        if re.search(bridge_pattern, content, re.DOTALL):
            _bt = bridges_text + '\n\n'
            content = re.sub(bridge_pattern, lambda m: _bt, content, flags=re.DOTALL)
        else:
            content += '\n\n' + bridges_text

    # Update last updated date
    content = re.sub(
        r'\*\*Last Updated:\*\* \d{4}-\d{2}-\d{2}',
        f'**Last Updated:** {date.today().isoformat()}',
        content
    )

    review_path.write_text(content, encoding='utf-8')

    # Record in DB
    db.insert_review_update(
        category=category,
        update_type='weekly',
        fronts_changed=len(fronts),
        summary=f"Weekly revision: {len(fronts)} fronts, {len(bridges)} bridges"
    )

    print(f"  [OK] Revised {review_path.name}: {len(fronts)} fronts, {len(bridges)} bridges")
    return True
