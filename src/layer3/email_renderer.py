"""
Layer 3: Email Renderer

Generates HTML emails for daily, weekly, and monthly reports.
All styles are inline (email clients strip <style> blocks).
"""

from datetime import date
from typing import Dict, List, Optional


# ============================================================================
# Style Constants - Professional Consulting Theme
# ============================================================================

COLORS = {
    # Backgrounds - Clean corporate
    'bg': '#F5F7FA',              # Light gray background
    'card_bg': '#FFFFFF',         # Pure white cards
    'accent_bg': '#F8F9FB',       # Very light gray accent
    'section_bg': '#FAFBFC',      # Section backgrounds

    # Borders and dividers - Structured
    'border': '#E1E4E8',          # Light gray border
    'border_strong': '#D1D5DB',   # Medium gray border

    # Text - Professional hierarchy
    'text': '#1F2937',            # Dark gray (not black)
    'text_muted': '#6B7280',      # Medium gray
    'text_light': '#9CA3AF',      # Light gray

    # Links - Professional blue
    'link': '#2563EB',            # Standard blue
    'link_hover': '#1D4ED8',      # Darker blue

    # Score indicators - Traffic light system
    'score_high': '#059669',      # Green (7+)
    'score_mid': '#D97706',       # Amber (5-6)
    'score_low': '#6B7280',       # Gray (<5)

    # Significance badges - Serious tones
    'badge_must_read': '#DC2626',      # Red
    'badge_changes': '#7C3AED',        # Purple
    'badge_discuss': '#2563EB',        # Blue

    # Front status - Corporate colors
    'status_emerging': '#0891B2',      # Teal
    'status_growing': '#059669',       # Green
    'status_stable': '#4F46E5',        # Indigo
    'status_declining': '#EA580C',     # Orange
    'status_merged': '#9333EA',        # Purple
    'status_split': '#DB2777',         # Pink

    # Header gradient - Navy to blue
    'gradient_start': '#1E3A8A',       # Navy blue
    'gradient_end': '#2563EB',         # Blue

    # Accents - Minimal, professional
    'accent_primary': '#2563EB',       # Blue
    'accent_secondary': '#4F46E5',     # Indigo
    'success': '#059669',              # Green
    'warning': '#D97706',              # Amber
    'danger': '#DC2626',               # Red
}


def _score_color(score: int) -> str:
    if score >= 7:
        return COLORS['score_high']
    elif score >= 5:
        return COLORS['score_mid']
    return COLORS['score_low']


def _status_color(status: str) -> str:
    return COLORS.get(f'status_{status}', COLORS['status_stable'])


def _density_label(density: float) -> str:
    if density >= 0.95:
        return "fully connected"
    elif density >= 0.6:
        return "dense"
    elif density >= 0.3:
        return "moderate"
    return "sparse"


def _front_short_id(front_id: str) -> str:
    if '_front_' in front_id:
        return front_id.split('_front_')[-1]
    return front_id[-4:]


# ============================================================================
# HTML Building Blocks
# ============================================================================

def _markdown_to_html(text: str) -> str:
    """Convert markdown formatting to HTML (bold, italic)."""
    import re

    # Convert **bold** to <strong>bold</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Convert __bold__ to <strong>bold</strong> (alternative syntax)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)

    # Convert *italic* to <em>italic</em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # Convert _italic_ to <em>italic</em> (alternative syntax)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)

    return text


def _wrap_email(title: str, body: str, date_str: str = None,
                issue_code: str = None) -> str:
    if date_str is None:
        date_str = date.today().isoformat()

    # Use provided issue_code, or fall back to ISO week number
    if issue_code is None:
        from datetime import datetime
        date_obj = datetime.fromisoformat(date_str)
        year = date_obj.year
        week = date_obj.isocalendar()[1]
        issue_code = f"#{week} of {year}"

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title>
</head>
<body style="margin:0; padding:0; background:{COLORS['bg']}; font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; color:{COLORS['text']}; font-size:14px; line-height:1.5;">
<div style="display:none; max-height:0; overflow:hidden; mso-hide:all; opacity:0; color:transparent;">Research Intelligence curated digest — Latest papers, research fronts, and framework evolution.</div>
<table width="100%" cellpadding="0" cellspacing="0" style="background:{COLORS['bg']};">
<tr><td align="center" style="padding:0;">
<table width="85%" cellpadding="0" cellspacing="0" style="width:85%;">

<tr><td style="background:linear-gradient(135deg, {COLORS['gradient_start']} 0%, {COLORS['gradient_end']} 100%); color:white; padding:20px 24px; border-radius:16px 16px 0 0;">
<table width="100%" cellpadding="0" cellspacing="0">
<tr>
<td style="width:48px; vertical-align:top;">
<div style="width:44px; height:44px; background:rgba(255,255,255,0.2); border-radius:12px; display:flex; align-items:center; justify-content:center; border:2px solid rgba(255,255,255,0.3);">
<span style="font-size:18px; font-weight:800; color:white; line-height:44px; text-align:center; display:block;">RI</span>
</div>
</td>
<td style="padding-left:14px; vertical-align:top;">
<h1 style="margin:0; font-size:20px; font-weight:700; letter-spacing:-0.01em; line-height:1.2;">{title}</h1>
<p style="margin:4px 0 0; font-size:11px; opacity:0.85; font-weight:500;">Issue {issue_code} · {date_str}</p>
</td>
</tr>
</table>
</td></tr>
<tr><td style="background:{COLORS['card_bg']}; padding:0;">
{body}
</td></tr>
<tr><td style="background:{COLORS['section_bg']}; padding:20px 24px; border-top:1px solid {COLORS['border']}; border-radius:0 0 16px 16px; text-align:center;">
<p style="margin:0 0 12px; font-size:11px; color:{COLORS['text_muted']}; font-weight:500;">Curated by Research Intelligence System</p>
<table cellpadding="0" cellspacing="0" role="presentation" style="margin:0 auto;">
<tr>
<td style="border-radius:12px; background:{COLORS['accent_primary']};">
<a href="#" style="display:inline-block; padding:10px 20px; font-size:12px; font-weight:700; color:#FFFFFF; text-decoration:none; border-radius:12px;">View Full Archive →</a>
</td>
</tr>
</table>
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>"""


def _section_header(title: str, subtitle: str = '') -> str:
    sub = f'<p style="margin:4px 0 0; font-size:11px; color:{COLORS["text_muted"]}; font-weight:500;">{subtitle}</p>' if subtitle else ''
    return f"""<tr><td style="padding:20px 24px 12px; border-top:2px solid {COLORS['accent_bg']}; background:{COLORS['section_bg']};">
  <h2 style="margin:0; font-size:15px; font-weight:700; color:{COLORS['text']}; text-transform:uppercase; letter-spacing:0.03em;">{title}</h2>
  {sub}
</td></tr>"""


def _paper_row(paper: dict, rank: int = 0) -> str:
    """Render a single paper as an HTML row with modern styling."""
    rel = paper.get('relevance', {})
    sig = paper.get('significance', {})
    m = rel.get('methodological', 0)
    p = rel.get('problem', 0)
    i = rel.get('inspirational', 0)

    # Pick ONE most important badge (modern design: limit visual noise)
    badge_html = ''
    if sig.get('must_read'):
        badge_html = f'<span style="background:{COLORS["badge_must_read"]}; color:white; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:600;">MUST-READ</span>'
    elif sig.get('changes_thinking'):
        badge_html = f'<span style="background:{COLORS["badge_changes"]}; color:white; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:600;">CHANGES THINKING</span>'
    elif sig.get('team_discussion'):
        badge_html = f'<span style="background:{COLORS["badge_discuss"]}; color:white; padding:4px 12px; border-radius:999px; font-size:10px; font-weight:600;">DISCUSS</span>'

    title = paper.get('title', 'Untitled')
    arxiv_id = paper.get('arxiv_id', '')
    arxiv_url = paper.get('arxiv_url', f'https://arxiv.org/abs/{arxiv_id}')
    pub_date = paper.get('published_date', '')[:10]
    affiliations = paper.get('affiliations', '')

    full_brief = _markdown_to_html(paper.get('brief', ''))
    if len(full_brief) > 300:
        brief_html = (
            f"{full_brief[:300]}..."
            f'<details style="margin-top:6px;">'
            f'<summary style="color:{COLORS["accent_primary"]}; cursor:pointer; font-weight:600; font-size:12px; list-style:none; display:block;">&#9654; Read more</summary>'
            f'<div style="padding-top:8px; border-top:1px solid {COLORS["border"]}; margin-top:6px;">{full_brief}</div>'
            f'</details>'
        )
    else:
        brief_html = full_brief

    # Modern rank badge
    rank_label = f'<span style="display:inline-block; background:{COLORS["accent_primary"]}; color:white; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; margin-right:8px;">#{rank}</span>' if rank else ''

    # M, P, I as muted chips
    mpi_chips = f'''<span style="background:{COLORS['accent_bg']}; color:{COLORS['text_muted']}; padding:2px 8px; border-radius:999px; font-weight:600; font-size:10px; margin-right:4px;">M={m}</span>
    <span style="background:{COLORS['accent_bg']}; color:{COLORS['text_muted']}; padding:2px 8px; border-radius:999px; font-weight:600; font-size:10px; margin-right:4px;">P={p}</span>
    <span style="background:{COLORS['accent_bg']}; color:{COLORS['text_muted']}; padding:2px 8px; border-radius:999px; font-weight:600; font-size:10px;">I={i}</span>'''

    return f"""<tr><td style="padding:16px 24px; border-bottom:1px solid {COLORS['border']}; background:{COLORS['card_bg']};">
  <div style="margin-bottom:8px;">
    {rank_label}<a href="{arxiv_url}" style="color:{COLORS['text']}; text-decoration:none; font-weight:700; font-size:16px; letter-spacing:-0.01em;">{title}</a>
  </div>
  <div style="margin-bottom:8px;">
    {badge_html}
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


# ============================================================================
# Daily Email
# ============================================================================

def _daily_summary_bar(total_new: int, total_must_read: int, categories_count: int) -> str:
    """Render daily summary bar with key metrics."""
    return f"""<tr><td style="background:{COLORS['card_bg']}; padding:20px 24px 12px;">
  <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid {COLORS['border']}; border-radius:14px; background:{COLORS['section_bg']};">
    <tr>
      <td style="padding:14px 18px; font-size:12px; color:{COLORS['text']}; font-weight:700; text-transform:uppercase; letter-spacing:0.03em; border-bottom:1px solid {COLORS['border']};">
        Today at a glance
      </td>
    </tr>
    <tr>
      <td style="padding:0 18px 14px;">
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td style="width:33%; padding:12px 0; text-align:center;">
              <div style="font-size:24px; font-weight:800; color:{COLORS['accent_primary']};">{total_must_read}</div>
              <div style="font-size:11px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.02em;">Must-reads</div>
            </td>
            <td style="width:33%; padding:12px 0; text-align:center; border-left:1px solid {COLORS['border']}; border-right:1px solid {COLORS['border']};">
              <div style="font-size:24px; font-weight:800; color:{COLORS['accent_primary']};">{total_new}</div>
              <div style="font-size:11px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.02em;">New papers</div>
            </td>
            <td style="width:33%; padding:12px 0; text-align:center;">
              <div style="font-size:24px; font-weight:800; color:{COLORS['accent_primary']};">{categories_count}</div>
              <div style="font-size:11px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.02em;">Categories</div>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</td></tr>"""


def render_daily_email(categories_data: List[dict], issue_code: str = None) -> str:
    """
    Render bi-daily email HTML with modern design.

    Args:
        categories_data: list of dicts from collect_daily_data()
        issue_code: issue identifier string, e.g. "Issue #5 of 2026".
                    If None, falls back to ISO week number.
    """
    body_parts = []

    # Calculate totals for summary bar
    total_new = sum(cat['stats']['new_count'] for cat in categories_data)
    total_must_read = sum(cat['stats']['must_read_count'] for cat in categories_data)
    categories_count = len(categories_data)

    # Add hero summary bar
    body_parts.append(_daily_summary_bar(total_new, total_must_read, categories_count))

    for cat_data in categories_data:
        category = cat_data['category']
        papers = cat_data['papers']
        stats = cat_data['stats']

        body_parts.append(_section_header(
            category,
            f"{stats['new_count']} new papers | {stats['must_read_count']} must-read | {stats['total_in_category']} total analyzed"
        ))

        if not papers:
            body_parts.append(f'<tr><td style="padding:12px 24px; color:{COLORS["text_muted"]};">No new papers this period.</td></tr>')
            continue

        for rank, paper in enumerate(papers, 1):
            body_parts.append(_paper_row(paper, rank))

    body = '<table width="100%" cellpadding="0" cellspacing="0">' + ''.join(body_parts) + '</table>'
    return _wrap_email(
        "Bi-Daily Research Intelligence Briefing",
        body,
        date.today().isoformat(),
        issue_code=issue_code
    )


# ============================================================================
# Weekly Email — with Research Landscape section
# ============================================================================

def _render_front_card(front: dict, front_methods: dict,
                       all_front_methods: dict) -> str:
    """Render a single research front card."""
    fid = front['front_id']
    short_id = _front_short_id(fid)
    status = front.get('status', 'unknown')
    size = front.get('size', 0)
    density = front.get('internal_density', 0)
    summary = front.get('summary', '')
    dom_methods = front.get('dominant_methods', [])
    dom_problems = front.get('dominant_problems', [])

    status_c = _status_color(status)
    status_badge = f'<span style="background:{status_c}; color:white; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:bold;">{status.upper()}</span>'

    # Unique vs shared methods
    my_methods = set(front_methods.get(fid, []))
    other_methods = set()
    for other_fid, other_m in all_front_methods.items():
        if other_fid != fid:
            other_methods.update(other_m)
    unique = sorted(my_methods - other_methods)
    shared = sorted(my_methods & other_methods)

    methods_html = ''
    if unique:
        methods_html += f'<div style="font-size:11px; margin-top:4px;"><b>Unique:</b> {", ".join(unique)}</div>'
    if shared:
        methods_html += f'<div style="font-size:11px; margin-top:2px;"><b>Shared:</b> {", ".join(shared)}</div>'

    # Paper list
    papers_html = ''
    for p in front.get('papers_detail', []):
        meth = p.get('methodology', {})
        core = meth.get('core_method', '?')[:50]
        llm_role = meth.get('llm_role', 'none')
        pid = p.get('arxiv_id', '')
        ptitle = p.get('title', '')[:60]
        papers_html += f'<div style="font-size:11px; padding:2px 0; border-bottom:1px solid #f0f0f0;"><a href="https://arxiv.org/abs/{pid}" style="color:{COLORS["link"]}; text-decoration:none;">{ptitle}</a> &mdash; <span style="color:{COLORS["text_muted"]};">{core} | LLM: {llm_role}</span></div>'

    dom_html = ''
    if dom_methods:
        dom_html += f'<span style="font-size:11px; color:{COLORS["text_muted"]};">Methods: {", ".join(dom_methods[:5])}</span>'
    if dom_problems:
        dom_html += f' <span style="font-size:11px; color:{COLORS["text_muted"]};">| Problems: {", ".join(dom_problems[:5])}</span>'

    summary_html = ''
    if summary:
        summary_html = f'<div style="margin-top:8px; font-size:13px; color:{COLORS["text"]}; line-height:1.4; background:#F8F9FA; padding:10px; border-radius:4px;">{summary[:500]}{"..." if len(summary) > 500 else ""}</div>'

    return f"""<tr><td style="padding:16px 24px; border-bottom:1px solid {COLORS['border']};">
  <div style="margin-bottom:6px;">
    <span style="font-size:15px; font-weight:bold;">Front {short_id}</span>
    &nbsp;{status_badge}&nbsp;
    <span style="font-size:12px; color:{COLORS['text_muted']};">{size} papers &mdash; density: {density:.2f} ({_density_label(density)})</span>
  </div>
  {dom_html}
  {methods_html}
  {summary_html}
  <div style="margin-top:8px;">
    <details>
      <summary style="font-size:12px; color:{COLORS['link']}; font-weight:bold; cursor:pointer; list-style:none; display:block;">&#9654; Papers in this front</summary>
      <div style="padding-left:8px; margin-top:4px;">
        {papers_html}
      </div>
    </details>
  </div>
</td></tr>"""


def _render_bridge_card(bridge: dict) -> str:
    """Render a bridge paper card."""
    paper = bridge.get('paper')
    pid = bridge.get('paper_id', '')
    score = bridge.get('bridge_score', 0)
    verdict = bridge.get('verdict', '?')
    home = _front_short_id(bridge.get('home_front_id', ''))
    connected = [_front_short_id(c) for c in bridge.get('connected_fronts', [])]

    verdict_colors = {
        'TRUE SYNTHESIS': '#2E7D32',
        'CITING BRIDGE': '#F57F17',
        'HOME-ANCHORED': '#1565C0',
        'STRUCTURAL': '#757575',
    }
    vc = verdict_colors.get(verdict, '#757575')

    title = ''
    brief = ''
    if paper:
        title = paper.get('title', pid)
        brief = paper.get('brief', '')[:200]
    else:
        title = pid

    return f"""<tr><td style="padding:10px 24px; border-bottom:1px solid {COLORS['border']};">
  <div>
    <a href="https://arxiv.org/abs/{pid}" style="color:{COLORS['link']}; text-decoration:none; font-weight:bold; font-size:13px;">{title}</a>
  </div>
  <div style="margin-top:4px; font-size:11px;">
    <span style="color:{vc}; font-weight:bold;">{verdict}</span>
    &mdash; score={score:.2f}
    &mdash; Front {home} &#8594; {', '.join('Front ' + c for c in connected)}
  </div>
  <div style="margin-top:4px; font-size:12px; color:{COLORS['text_muted']};">{brief}{'...' if len(paper.get('brief', '') if paper else '') > 200 else ''}</div>
</td></tr>"""


def _render_method_overlap_matrix(fronts: List[dict],
                                  front_methods: dict,
                                  method_overlap: dict) -> str:
    """Render method overlap matrix as HTML table."""
    if len(fronts) < 2:
        return ''

    short_ids = [_front_short_id(f['front_id']) for f in fronts]
    fids = [f['front_id'] for f in fronts]

    header = '<th style="padding:4px 8px; font-size:11px; background:#F0F0F0;"></th>'
    for sid in short_ids:
        header += f'<th style="padding:4px 8px; font-size:11px; background:#F0F0F0; text-align:center;">F{sid}</th>'

    rows = []
    for i, fi in enumerate(fids):
        cells = f'<td style="padding:4px 8px; font-size:11px; font-weight:bold; background:#F0F0F0;">F{short_ids[i]}</td>'
        for j, fj in enumerate(fids):
            if i == j:
                cells += f'<td style="padding:4px 8px; text-align:center; font-size:11px; color:{COLORS["text_muted"]};">&mdash;</td>'
            else:
                key = f"{fi}|{fj}" if f"{fi}|{fj}" in method_overlap else f"{fj}|{fi}"
                shared = method_overlap.get(key, [])
                count = len(shared)
                bg = '#E8F5E9' if count > 0 else '#FAFAFA'
                title_attr = f' title="{", ".join(shared)}"' if shared else ''
                cells += f'<td style="padding:4px 8px; text-align:center; font-size:11px; background:{bg};"{title_attr}>{count}</td>'
        rows.append(f'<tr>{cells}</tr>')

    return f"""<tr><td style="padding:16px 24px;">
  <div style="font-size:13px; font-weight:bold; margin-bottom:8px;">Method Overlap Between Fronts</div>
  <table cellpadding="0" cellspacing="1" style="border-collapse:collapse; border:1px solid {COLORS['border']};">
    <tr>{header}</tr>
    {''.join(rows)}
  </table>
  <div style="font-size:10px; color:{COLORS['text_muted']}; margin-top:4px;">Hover cells for shared method names</div>
</td></tr>"""


def render_weekly_email(categories_data: List[dict],
                        graph_images: Optional[Dict[str, str]] = None) -> str:
    """
    Render weekly email HTML with Research Landscape section.

    Args:
        categories_data: list of dicts from collect_weekly_data()
        graph_images: dict mapping category -> base64 PNG string
    """
    if graph_images is None:
        graph_images = {}

    body_parts = []

    for cat_data in categories_data:
        category = cat_data['category']
        papers = cat_data['papers']
        fronts = cat_data.get('fronts', [])
        bridges = cat_data.get('bridges', [])
        front_methods = cat_data.get('front_methods', {})
        method_overlap = cat_data.get('method_overlap', {})
        stats = cat_data['stats']

        # ── Category Header ──
        body_parts.append(_section_header(
            category,
            f"{stats['new_count']} new &mdash; {stats['must_read_count']} must-read &mdash; "
            f"{stats['fronts_count']} fronts &mdash; {stats['bridges_count']} bridges &mdash; "
            f"{stats['total_in_category']} total"
        ))

        # ── Part A: Top Papers This Week ──
        body_parts.append(f'<tr><td style="padding:12px 24px 4px;"><h3 style="margin:0; font-size:14px; color:#37474F;">Top Papers This Week</h3></td></tr>')

        if not papers:
            body_parts.append(f'<tr><td style="padding:8px 24px; color:{COLORS["text_muted"]}; font-size:13px;">No new papers this week.</td></tr>')
        else:
            for rank, paper in enumerate(papers[:10], 1):
                body_parts.append(_paper_row(paper, rank))

        # ── Part B: Research Landscape ──
        if fronts:
            body_parts.append(f'<tr><td style="padding:20px 24px 4px; border-top:2px solid #E8EAF6;"><h3 style="margin:0; font-size:14px; color:#1A237E;">Research Landscape</h3><p style="margin:2px 0 0; font-size:11px; color:{COLORS["text_muted"]};">Layer 2 bibliometric analysis &mdash; research fronts, bridge papers, method landscape</p></td></tr>')

            # Front cards
            for front in fronts:
                body_parts.append(_render_front_card(front, front_methods, front_methods))

            # Method overlap matrix
            if len(fronts) > 1:
                body_parts.append(_render_method_overlap_matrix(fronts, front_methods, method_overlap))

            # Bridge papers
            if bridges:
                body_parts.append(f'<tr><td style="padding:16px 24px 4px;"><div style="font-size:13px; font-weight:bold;">Bridge Papers</div><div style="font-size:11px; color:{COLORS["text_muted"]};">Papers connecting multiple research fronts</div></td></tr>')
                for bridge in bridges:
                    body_parts.append(_render_bridge_card(bridge))

            # Embedded graph
            b64_img = graph_images.get(category)
            if b64_img:
                body_parts.append(f'<tr><td style="padding:16px 24px; text-align:center;"><img src="data:image/png;base64,{b64_img}" style="max-width:100%; border:1px solid {COLORS["border"]}; border-radius:4px;" alt="Front network graph"></td></tr>')

    body = '\n<table width="100%" cellpadding="0" cellspacing="0">\n' + '\n'.join(body_parts) + '\n</table>'
    return _wrap_email(
        "Weekly Research Intelligence Report",
        body,
        date.today().isoformat()
    )


# ============================================================================
# Monthly Email
# ============================================================================

def render_monthly_email(categories_data: List[dict],
                         review_texts: Optional[Dict[str, str]] = None,
                         graph_images: Optional[Dict[str, str]] = None) -> str:
    """
    Render monthly digest email.

    Args:
        categories_data: list of dicts from collect_monthly_data()
        review_texts: dict mapping category -> LLM-generated monthly review text
        graph_images: dict mapping category -> base64 PNG string
    """
    if review_texts is None:
        review_texts = {}
    if graph_images is None:
        graph_images = {}

    body_parts = []

    for cat_data in categories_data:
        category = cat_data['category']
        stats = cat_data['stats']
        fronts = cat_data.get('fronts', [])

        body_parts.append(_section_header(
            category,
            f"{stats['total_in_category']} papers analyzed &mdash; {stats['fronts_count']} active fronts"
        ))

        # LLM-generated review
        review = review_texts.get(category, '')
        if review:
            paragraphs = review.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if para:
                    body_parts.append(f'<tr><td style="padding:8px 24px; font-size:13px; line-height:1.5;">{para}</td></tr>')

        # Graph
        b64_img = graph_images.get(category)
        if b64_img:
            body_parts.append(f'<tr><td style="padding:16px 24px; text-align:center;"><img src="data:image/png;base64,{b64_img}" style="max-width:100%; border:1px solid {COLORS["border"]}; border-radius:4px;" alt="Front network"></td></tr>')

        # Top must-reads from full history
        all_papers = cat_data.get('all_papers', cat_data.get('papers', []))
        must_reads = [p for p in all_papers if p.get('significance', {}).get('must_read')]
        if must_reads:
            body_parts.append(f'<tr><td style="padding:12px 24px;"><h3 style="margin:0; font-size:14px; color:#37474F;">Must-Read Papers</h3></td></tr>')
            for rank, paper in enumerate(must_reads[:10], 1):
                body_parts.append(_paper_row(paper, rank))

    body = '\n<table width="100%" cellpadding="0" cellspacing="0">\n' + '\n'.join(body_parts) + '\n</table>'
    return _wrap_email(
        "Monthly Research Intelligence Digest",
        body,
        date.today().isoformat()
    )
