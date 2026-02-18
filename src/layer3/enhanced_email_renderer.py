"""
Layer 3: Enhanced Email Renderer

Generates rich weekly emails with:
1. Top Priority Papers (smart ranked)
2. Research Front Analysis (with affiliation impact)
3. Important Bridge Papers
4. Framework Genealogy
5. Optional graphs
"""

from datetime import date
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from layer3.email_renderer import (
    COLORS, _score_color, _status_color, _density_label,
    _front_short_id, _wrap_email, _section_header, _markdown_to_html
)


def _hero_summary_bar(stats: dict, priority_count: int, fronts_count: int, editors_note: str = '') -> str:
    """Render hero summary bar with key metrics and editor's note."""
    must_read = stats.get('must_read_count', 0)
    new_papers = stats.get('new_count', 0)

    note_html = ''
    if editors_note:
        note_html = f'''<div style="margin-top:10px; padding-top:10px; border-top:1px solid {COLORS['border']}; font-size:12px; color:{COLORS['text']};">
          <b>This week's theme:</b> {editors_note}
        </div>'''

    return f"""<tr><td style="background:{COLORS['card_bg']}; padding:20px 24px 12px;">
  <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid {COLORS['border']}; border-radius:14px; background:{COLORS['section_bg']};">
    <tr>
      <td style="padding:14px 18px; font-size:12px; color:{COLORS['text']}; font-weight:700; text-transform:uppercase; letter-spacing:0.03em; border-bottom:1px solid {COLORS['border']};">
        This week at a glance
      </td>
    </tr>
    <tr>
      <td style="padding:0 18px 14px;">
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td style="width:33%; padding:12px 0; text-align:center;">
              <div style="font-size:24px; font-weight:800; color:{COLORS['accent_primary']};">{must_read}</div>
              <div style="font-size:11px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.02em;">Must-reads</div>
            </td>
            <td style="width:33%; padding:12px 0; text-align:center; border-left:1px solid {COLORS['border']}; border-right:1px solid {COLORS['border']};">
              <div style="font-size:24px; font-weight:800; color:{COLORS['accent_primary']};">{new_papers}</div>
              <div style="font-size:11px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.02em;">New papers</div>
            </td>
            <td style="width:33%; padding:12px 0; text-align:center;">
              <div style="font-size:24px; font-weight:800; color:{COLORS['accent_primary']};">{fronts_count}</div>
              <div style="font-size:11px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.02em;">Active fronts</div>
            </td>
          </tr>
        </table>
        {note_html}
      </td>
    </tr>
  </table>
</td></tr>"""


def _priority_paper_row(item: dict, rank: int) -> str:
    """Render a priority-ranked paper with score and reasons."""
    paper = item['paper']
    score = item['score']
    reasons = item.get('reasons', [])

    rel = paper.get('relevance', {})
    sig = paper.get('significance', {})
    m = rel.get('methodological', 0)
    p = rel.get('problem', 0)
    i = rel.get('inspirational', 0)

    # Badges - Limit to max 2: priority score + one most important reason
    # Priority score indicator
    score_color = COLORS['score_high'] if score >= 7 else (COLORS['score_mid'] if score >= 5 else COLORS['score_low'])
    score_badge = f'<span style="background:{score_color}; color:white; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:700;">PRIORITY {score:.1f}/10</span>'

    # Pick ONE most important reason badge
    secondary_badge = ''
    for reason in reasons:
        if 'MUST-READ' in reason:
            secondary_badge = f'<span style="background:{COLORS["badge_must_read"]}; color:white; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:600;">MUST-READ</span>'
            break
        elif 'Changes thinking' in reason:
            secondary_badge = f'<span style="background:{COLORS["badge_changes"]}; color:white; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:600;">CHANGES THINKING</span>'
            break
        elif 'Emerging' in reason or 'Growing' in reason:
            secondary_badge = f'<span style="background:{COLORS["status_emerging"]}; color:white; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:600;">EMERGING FRONT</span>'
            break
        elif 'Bridge' in reason:
            secondary_badge = f'<span style="background:{COLORS["badge_discuss"]}; color:white; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:600;">BRIDGE</span>'
            break

    title = paper.get('title', 'Untitled')
    arxiv_id = paper.get('arxiv_id', '')
    arxiv_url = paper.get('arxiv_url', f'https://arxiv.org/abs/{arxiv_id}')
    pub_date = paper.get('published_date', '')[:10]
    affiliations = paper.get('affiliations', '')

    # Brief — always show in full (<details> is stripped by Gmail)
    brief_html = _markdown_to_html(paper.get('brief', ''))

    # MPI scores as muted chips
    mpi_chips = f'''<span style="background:{COLORS['accent_bg']}; color:{COLORS['text_muted']}; padding:2px 8px; border-radius:999px; font-weight:600; font-size:10px; margin-right:4px;">M={m}</span>
    <span style="background:{COLORS['accent_bg']}; color:{COLORS['text_muted']}; padding:2px 8px; border-radius:999px; font-weight:600; font-size:10px; margin-right:4px;">P={p}</span>
    <span style="background:{COLORS['accent_bg']}; color:{COLORS['text_muted']}; padding:2px 8px; border-radius:999px; font-weight:600; font-size:10px;">I={i}</span>'''

    return f"""<tr><td style="padding:16px 24px; border-bottom:1px solid {COLORS['border']}; background:{COLORS['card_bg']};">
  <div style="margin-bottom:8px;">
    <span style="display:inline-block; background:{COLORS['accent_primary']}; color:white; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; margin-right:8px;">#{rank}</span>
    <a href="{arxiv_url}" style="color:{COLORS['text']}; text-decoration:none; font-weight:700; font-size:16px; letter-spacing:-0.01em;">{title}</a>
  </div>
  <div style="margin-bottom:8px;">
    {score_badge} {secondary_badge}
  </div>
  <div style="margin-bottom:6px; font-size:11px; color:{COLORS['text_muted']};">
    {pub_date}{(' | ' + affiliations) if affiliations else ''} | <a href="{arxiv_url}" style="color:{COLORS['text_muted']}; text-decoration:none;">{arxiv_id}</a>
  </div>
  <div style="margin-bottom:8px;">
    {mpi_chips}
  </div>
  <div style="font-size:13px; color:{COLORS['text']}; line-height:1.6; padding:12px; background:{COLORS['accent_bg']}; border-radius:12px; border-left:3px solid {COLORS['accent_primary']};">
    {brief_html}
  </div>
</td></tr>"""


def _front_card_with_affiliations(front: dict, aff_info: Optional[dict]) -> str:
    """Render a front card with affiliation impact."""
    fid = front['front_id']
    short_id = _front_short_id(fid)
    status = front.get('status', 'unknown')
    size = front.get('size', 0)
    density = front.get('internal_density', 0)
    summary = front.get('summary', '')
    dom_methods = front.get('dominant_methods', [])

    # Use LLM-generated name; fall back to dominant methods, never "Front N"
    raw_name = (front.get('name') or '').strip()
    if not raw_name or raw_name.startswith('['):
        raw_name = ' · '.join(dom_methods[:3]) if dom_methods else f'Research Cluster {short_id}'
    name = raw_name

    status_c = _status_color(status)
    status_badge = f'<span style="background:{status_c}; color:white; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:700; text-transform:uppercase; display:inline-block;">{status.upper()}</span>'
    density_badge = f'<span style="background:{COLORS["accent_bg"]}; color:{COLORS["text"]}; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:600; display:inline-block;">Density: {density:.2f}</span>'

    # Compact institutions — single inline line, no card wrapper
    aff_html = ''
    if aff_info:
        top_affs = aff_info.get('top_affiliations', [])[:4]
        if top_affs:
            parts = []
            for aff, count in top_affs:
                pct = (count / size * 100) if size > 0 else 0
                # Shorten very long names: keep first two words
                short_aff = ' '.join(aff.split()[:3])
                if short_aff != aff:
                    short_aff += '.'
                parts.append(f'{short_aff} {pct:.0f}%')
            aff_line = ' &nbsp;·&nbsp; '.join(parts)
            aff_html = (
                f'<div style="margin-top:6px; font-size:10px; color:{COLORS["text_muted"]};">'
                f'<span style="font-weight:700; text-transform:uppercase; letter-spacing:0.02em; margin-right:4px;">Inst:</span>'
                f'{aff_line}</div>'
            )

    # Methods — compact pill tags
    methods_html = ''
    if dom_methods:
        method_tags = [
            f'<span style="display:inline-block; background:{COLORS["accent_bg"]}; color:{COLORS["accent_primary"]}; '
            f'padding:3px 10px; border-radius:999px; margin:2px; font-size:10px; font-weight:600;">{m}</span>'
            for m in dom_methods[:8]
        ]
        methods_html = (
            f'<div style="margin-top:10px; line-height:1.8;">'
            f'<span style="font-size:11px; color:{COLORS["text_muted"]}; font-weight:600; '
            f'text-transform:uppercase; letter-spacing:0.02em; margin-right:6px;">Methods</span>'
            f'{" ".join(method_tags)}</div>'
        )

    # Summary — always show in full (<details> is stripped by Gmail)
    summary_html = ''
    if summary and not summary.startswith('['):
        paragraphs = [p.strip() for p in summary.replace('\\n\\n', '\n\n').split('\n\n') if p.strip()]
        all_paras = ''.join(
            f'<p style="margin:{"0" if i == 0 else "8px"} 0 0;">{_markdown_to_html(p)}</p>'
            for i, p in enumerate(paragraphs)
        )
        summary_html = (
            f'<div style="margin-top:10px; font-size:13px; line-height:1.6; color:{COLORS["text"]}; '
            f'padding:12px; background:{COLORS["accent_bg"]}; border-radius:12px; '
            f'border-left:3px solid {COLORS["accent_secondary"]};">{all_paras}</div>'
        )

    papers_count = f'<span style="background:{COLORS["accent_bg"]}; color:{COLORS["text"]}; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:600; display:inline-block;">{size} papers</span>'

    # Collapsed paper list
    papers_detail = front.get('papers_detail', [])
    papers_html = ''
    if papers_detail:
        rows = []
        for p in papers_detail:
            pid = p.get('arxiv_id', '')
            title = p.get('title', pid)
            pub = (p.get('published_date') or '')[:7]
            rel = p.get('relevance') or {}
            m_score = rel.get('methodological', '')
            must = (p.get('significance') or {}).get('must_read', False)
            must_badge = (
                f' <span style="background:{COLORS["score_high"]}; color:white; '
                f'padding:1px 6px; border-radius:999px; font-size:9px; font-weight:700;">★</span>'
                if must else ''
            )
            score_str = (
                f' <span style="color:{COLORS["text_muted"]}; font-size:10px;">M:{m_score}</span>'
                if m_score != '' else ''
            )
            rows.append(
                f'<li style="padding:5px 0; border-bottom:1px solid {COLORS["border"]}; list-style:none;">'
                f'<a href="https://arxiv.org/abs/{pid}" style="color:{COLORS["accent_primary"]}; '
                f'text-decoration:none; font-size:12px; font-weight:500;">{title}</a>'
                f'{must_badge}{score_str}'
                f' <span style="color:{COLORS["text_muted"]}; font-size:10px;">· {pub}</span>'
                f'</li>'
            )
        papers_html = (
            f'<div style="margin-top:10px; font-size:11px; color:{COLORS["accent_primary"]}; font-weight:600;">'
            f'{len(papers_detail)} papers in this front</div>'
            f'<ul style="margin:4px 0 0; padding:0; border-top:1px solid {COLORS["border"]};">'
            + ''.join(rows)
            + '</ul>'
        )

    return f"""<tr><td style="padding:16px 24px; border-bottom:1px solid {COLORS['border']}; background:{COLORS['card_bg']};">
  <div>
    <h3 style="margin:0; font-size:16px; font-weight:700; color:{COLORS['text']}; letter-spacing:-0.01em;">{name}</h3>
  </div>
  <div style="margin-top:8px; line-height:1.8;">
    {status_badge} {density_badge} {papers_count}
  </div>
  {methods_html}
  {aff_html}
  {summary_html}
  {papers_html}
</td></tr>"""


def _bridge_paper_row(bridge: dict) -> str:
    """Render a bridge paper row."""
    paper = bridge.get('paper')
    if not paper:
        return ''

    pid = bridge.get('paper_id', '')
    score = bridge.get('bridge_score', 0)
    verdict = bridge.get('verdict', 'STRUCTURAL')
    home_fid = _front_short_id(bridge.get('home_front_id', ''))
    connected = [_front_short_id(c) for c in bridge.get('connected_fronts', [])]

    title = paper.get('title', 'Untitled')
    brief = paper.get('brief', '')[:200]
    pub_date = paper.get('published_date', '')[:10]

    # Limit to ONE badge - just the verdict (most important)
    verdict_colors = {
        'TRUE SYNTHESIS': COLORS['score_high'],
        'CITING BRIDGE': COLORS['score_mid'],
        'HOME-ANCHORED': COLORS['badge_discuss'],
        'STRUCTURAL': COLORS['text_muted'],
    }
    verdict_c = verdict_colors.get(verdict, COLORS['text_muted'])
    verdict_badge = f'<span style="background:{verdict_c}; color:white; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:600; text-transform:uppercase;">{verdict}</span>'

    # Connection info as muted chip
    connections = f'<span style="background:{COLORS["accent_bg"]}; color:{COLORS["text_muted"]}; padding:3px 10px; border-radius:999px; font-size:10px;">Front {home_fid} → {", ".join("Front " + c for c in connected)}</span>'

    return f"""<tr><td style="padding:14px 24px; border-bottom:1px solid {COLORS['border']};">
  <div style="margin-bottom:6px;">
    <a href="https://arxiv.org/abs/{pid}" style="color:{COLORS['text']}; text-decoration:none; font-weight:600; font-size:14px; letter-spacing:-0.01em;">{title}</a>
  </div>
  <div style="margin-bottom:6px;">
    {verdict_badge} {connections}
  </div>
  <div style="margin-bottom:6px; font-size:11px; color:{COLORS['text_muted']};">
    {pub_date} · {pid}
  </div>
  <div style="font-size:13px; color:{COLORS['text']}; line-height:1.5;">
    {brief}{'...' if len(paper.get('brief', '')) > 200 else ''}
  </div>
</td></tr>"""


def _genealogy_section(genealogy_data: dict) -> str:
    """Render framework genealogy section with word cloud."""
    from layer3.framework_wordcloud import generate_framework_wordcloud_html

    summary = genealogy_data.get('summary', {})
    active = genealogy_data.get('active_last_30d', [])
    lineages = genealogy_data.get('lineages', {})

    total = summary.get('total_frameworks', 0)
    roots = summary.get('root_frameworks', 0)
    active_count = summary.get('active_last_30d', 0)

    stats_html = f'''<div style="padding:12px 24px; background:{COLORS['accent_bg']}; border-radius:12px 12px 0 0; border:1px solid {COLORS['border']}; border-bottom:none;">
  <div style="font-size:11px; color:{COLORS['text']};">
    <b>{total}</b> frameworks tracked · <b>{roots}</b> root frameworks · <b>{active_count}</b> active (last 30 days)
  </div>
</div>'''

    # Word cloud (replaces text tree)
    wordcloud_html = ''
    if lineages:
        wordcloud_img = generate_framework_wordcloud_html(lineages, active, width=800, height=400)
        wordcloud_html = f'''<div style="padding:16px 24px; background:{COLORS['card_bg']}; text-align:center; border:1px solid {COLORS['border']}; border-top:none; border-radius:0 0 12px 12px;">
  <div style="font-size:10px; margin-bottom:10px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.03em;">Framework landscape (size = paper count, color = must-read ratio)</div>
  {wordcloud_img}
  <div style="margin-top:10px; font-size:9px; color:{COLORS['text_muted']};">
    <span style="color:#1A5F7A;">■</span> Active + Must-read &nbsp;
    <span style="color:#64B5CD;">■</span> Active &nbsp;
    <span style="color:#2E7D32;">■</span> Inactive + Must-read &nbsp;
    <span style="color:#A5D6A7;">■</span> Inactive
  </div>
</div>'''

    return f'''<tr><td style="padding:0;">
{stats_html}
{wordcloud_html}
</td></tr>'''


def render_enhanced_weekly_email(data: dict) -> str:
    """
    Render enhanced weekly email with prioritization, genealogy, and affiliations.

    Args:
        data: Dict from collect_enhanced_weekly_data()

    Email structure:
        1. Top Priority Papers (smart ranked)
        2. Research Front Analysis (with affiliation impact)
        3. Important Bridge Papers
        4. Framework Genealogy
    """
    category = data['category']
    stats = data['stats']
    priority_papers = data.get('priority_papers', [])
    fronts = data.get('fronts', [])
    bridges = data.get('bridges', [])
    genealogy = data.get('framework_genealogy', {})
    aff_analysis = data.get('affiliation_analysis', {})

    body_parts = []

    # === HERO SUMMARY BAR ===
    editors_note = "Concept-structured search is outperforming brute code mutation across multiple optimization domains."
    body_parts.append(_hero_summary_bar(stats, len(priority_papers), len(fronts), editors_note))

    # === SECTION 1: TOP PRIORITY PAPERS ===
    body_parts.append(_section_header(
        'Top Priority Papers',
        f'{len(priority_papers)} must-read papers this week (ranked by significance, recency, and impact)'
    ))

    if priority_papers:
        for item in priority_papers:
            body_parts.append(_priority_paper_row(item, item['rank']))
    else:
        body_parts.append(f'<tr><td style="padding:12px 24px; color:{COLORS["text_muted"]};">No priority papers this week.</td></tr>')

    # === SECTION 2: RESEARCH FRONTS ===
    body_parts.append(_section_header(
        'Research Front Landscape',
        f'{len(fronts)} active fronts | {stats.get("new_count", 0)} new papers'
    ))

    # Sort fronts by status priority: emerging > growing > stable > declining
    status_order = {'emerging': 0, 'growing': 1, 'stable': 2, 'declining': 3}
    fronts_sorted = sorted(fronts, key=lambda f: (status_order.get(f.get('status', 'stable'), 2), -f.get('size', 0)))

    per_front_affs = aff_analysis.get('per_front', {})
    for front in fronts_sorted:
        fid = front['front_id']
        aff_info = per_front_affs.get(fid)
        body_parts.append(_front_card_with_affiliations(front, aff_info))

    # === SECTION 3: BRIDGE PAPERS ===
    if bridges:
        body_parts.append(_section_header(
            'Cross-Front Bridge Papers',
            f'{len(bridges)} papers connecting multiple research fronts'
        ))

        for bridge in bridges[:5]:  # Top 5 bridges
            body_parts.append(_bridge_paper_row(bridge))

    # === SECTION 4: FRAMEWORK GENEALOGY ===
    if genealogy and genealogy.get('summary', {}).get('total_frameworks', 0) > 0:
        body_parts.append(_section_header(
            'Framework Genealogy',
            'Tracking research lineages and framework evolution'
        ))
        body_parts.append(_genealogy_section(genealogy))

    body = '<table width="100%" cellpadding="0" cellspacing="0">' + ''.join(body_parts) + '</table>'

    return _wrap_email(
        f'Weekly Research Intelligence — {category}',
        body,
        date.today().isoformat()
    )
