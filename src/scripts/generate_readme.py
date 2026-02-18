#!/usr/bin/env python3
"""
Generate enriched README.md and docs/index.md from:
  - docs/or-llm-daily.json  (paper list, pipe-delimited strings)
  - src/db/research_intelligence.db  (L1 scores/briefs/affiliations, L2 fronts)

This is the ONLY script that writes README.md and docs/index.md.
Call it from all three workflows after their respective final analysis step.

Usage:
    python src/scripts/generate_readme.py
"""

import json
import re
import sys
import datetime
from pathlib import Path

# Reach repo root so imports work regardless of cwd
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from db.database import Database

# ── Paths ────────────────────────────────────────────────────────────────────

JSON_README  = REPO_ROOT / "docs" / "or-llm-daily.json"
JSON_GITPAGE = REPO_ROOT / "docs" / "or-llm-daily-web.json"
README_PATH  = REPO_ROOT / "README.md"
INDEX_PATH   = REPO_ROOT / "docs" / "index.md"

TOP_N = 5            # papers shown in Most Recent and Best Papers
MAX_FULL_LIST = 500  # safety cap for the full collapsible list

# ── Category display order ────────────────────────────────────────────────────

CATEGORY_ORDER = [
    "LLMs for Algorithm Design",
    "Generative AI for OR",
    "OR for Generative AI",
]

# ── Status emoji mapping (L2 research front status) ──────────────────────────

STATUS_EMOJI = {
    "emerging":  "🚀",
    "growing":   "📈",
    "stable":    "✅",
    "declining": "📉",
    "merged":    "🔀",
    "split":     "🔀",
    "new":       "🆕",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_id(arxiv_id: str) -> str:
    s = arxiv_id.strip()
    # Strip markdown link format: [2602.04529](http://arxiv.org/abs/...)
    m = re.match(r'\[([^\]]+)\]', s)
    if m:
        s = m.group(1)
    return re.sub(r'v\d+$', '', s)


def _readable_tag(tag: str) -> str:
    """Convert snake_case internal tag to human-readable Title Case."""
    return tag.replace("_", " ").title()


def _parse(s):
    """Return (date, title, authors, affiliation, venue, arxiv_id, code).

    Accepts either:
    - a dict from the `papers` DB table (preferred)
    - a pipe-delimited string from docs/or-llm-daily.json (fallback)
    """
    if isinstance(s, dict):
        code_url = s.get('code_url', '') or ''
        code = f"**[link]({code_url})**" if code_url else "null"
        return (
            s.get('date', ''),
            s.get('title', ''),
            s.get('authors', ''),
            s.get('affiliation', '') or '',
            s.get('venue', '') or '',
            s.get('arxiv_id', ''),  # already clean in DB
            code,
        )
    # Legacy pipe-delimited string from JSON
    parts = str(s).split("|")
    if len(parts) >= 9:
        date, title, authors = parts[1].strip(), parts[2].strip(), parts[3].strip()
        affiliation = parts[4].strip()
        venue       = parts[5].strip()
        arxiv_id    = parts[6].strip()
        code        = parts[7].strip()
    elif len(parts) >= 8:
        date, title, authors = parts[1].strip(), parts[2].strip(), parts[3].strip()
        affiliation = ""
        venue       = parts[4].strip()
        arxiv_id    = parts[5].strip()
        code        = parts[6].strip()
    else:
        date, title, authors = parts[1].strip(), parts[2].strip(), parts[3].strip()
        affiliation = ""
        venue       = ""
        arxiv_id    = parts[4].strip()
        code        = parts[5].strip()
    return date, title, authors, affiliation, venue, _clean_id(arxiv_id), code


def _date_key(s: str) -> str:
    m = re.search(r'(\d{4}[-.]?\d{2}[-.]?\d{2})', s)
    return m.group(1).replace('.', '-') if m else "0000-00-00"


def _plain_title(title: str) -> str:
    """Strip markdown bold/link syntax to get plain text."""
    t = re.sub(r'\*+', '', title).strip()
    m = re.match(r'\[([^\]]+)\]', t)
    return m.group(1) if m else t


def _title_cell(title: str, brief: str) -> str:
    """Wrap title in collapsible <details> if brief is available."""
    plain = _plain_title(title)
    if brief:
        safe = brief.replace("|", "&#124;").replace("\n", " ")
        return f"<details><summary>**{plain}**</summary>{safe}</details>"
    return f"**{plain}**"


def _resolve_code_url(code: str, db_code_url: str = "") -> str:
    """Extract a raw code URL from db_code_url or a pipe-string code field."""
    url = db_code_url or ""
    if not url:
        m = re.search(r'\[link\]\((.+?)\)', code)
        if m:
            url = m.group(1)
        elif code.strip().startswith("http"):
            url = code.strip()
    return url


def _links_cell(aid: str, code: str, db_code_url: str = "") -> str:
    """Markdown link version — for use in markdown tables (Most Recent, Best Papers)."""
    url = _resolve_code_url(code, db_code_url)
    pdf_link = f"[pdf](http://arxiv.org/abs/{aid})"
    code_part = f"[code]({url})" if url else "code"
    return f"{pdf_link} / {code_part}"


def _links_cell_html(aid: str, code: str, db_code_url: str = "") -> str:
    """HTML <a> tag version — for use inside HTML tables (Full list)."""
    url = _resolve_code_url(code, db_code_url)
    pdf_part = f'<a href="http://arxiv.org/abs/{aid}">pdf</a>'
    code_part = f'<a href="{url}">code</a>' if url else "code"
    return f"{pdf_part} / {code_part}"


def _is_visible(info: dict) -> bool:
    """True if a paper should appear in the README.

    - Not yet L1-analyzed (info is empty/None): show it, we don't know yet.
    - L1-analyzed and is_relevant=1: show it.
    - L1-analyzed and is_relevant=0: hide it (failed second-stage filter).
    """
    if not info:
        return True
    return bool(info.get("is_relevant", 1))


def _title_cell_html(title: str, brief: str) -> str:
    """Like _title_cell but uses <strong> instead of ** (for HTML table cells)."""
    plain = _plain_title(title)
    if brief:
        safe = brief.replace('"', "&quot;").replace("\n", " ")
        return (f'<details><summary><strong>{plain}</strong></summary>'
                f'{safe}</details>')
    return f"<strong>{plain}</strong>"

# ── DB queries ────────────────────────────────────────────────────────────────

def _load_l1(db: Database, category: str) -> dict:
    """Return {arxiv_id: {score, brief, affiliations, venue, code_url, is_relevant}} for all analyzed papers."""
    rows = db.fetchall(
        """SELECT arxiv_id, relevance, significance, brief, affiliations, venue, artifacts, is_relevant
           FROM paper_analyses
           WHERE category = ?""",
        (category,)
    )
    result = {}
    for row in rows:
        aid = _clean_id(row["arxiv_id"])
        try:
            rel = json.loads(row["relevance"] or "{}")
        except Exception:
            rel = {}
        try:
            artifacts = json.loads(row["artifacts"] or "{}")
        except Exception:
            artifacts = {}
        score = (rel.get("methodological", 0)
                 + rel.get("problem", 0)
                 + rel.get("inspirational", 0))
        result[aid] = {
            "score":        score,
            "brief":        (row["brief"] or "").strip(),
            "affiliations": (row["affiliations"] or "").strip(),
            "venue":        (row["venue"] or "").strip(),
            "code_url":     (artifacts.get("code_url") or "").strip(),
            "is_relevant":  row["is_relevant"],  # 1 = pass, 0 = filtered out
        }
    return result


def _load_fronts(db: Database, category: str) -> list:
    """Return front dicts for the latest L2 snapshot of this category."""
    row = db.fetchone(
        "SELECT MAX(snapshot_date) as d FROM research_fronts WHERE category = ?",
        (category,)
    )
    if not row or not row["d"]:
        return []
    snapshot = row["d"]
    rows = db.fetchall(
        """SELECT name, status, size, dominant_methods, dominant_problems, core_papers
           FROM research_fronts
           WHERE category = ? AND snapshot_date = ?
           ORDER BY size DESC""",
        (category, snapshot)
    )
    fronts = []
    for r in rows:
        try:
            methods = json.loads(r["dominant_methods"] or "[]")[:3]
        except Exception:
            methods = []
        try:
            problems = json.loads(r["dominant_problems"] or "[]")[:2]
        except Exception:
            problems = []
        try:
            core_papers = json.loads(r["core_papers"] or "[]")
        except Exception:
            core_papers = []
        fronts.append({
            "name":        r["name"] or "Unnamed front",
            "status":      r["status"] or "stable",
            "size":        r["size"] or 0,
            "methods":     methods,
            "problems":    problems,
            "core_papers": core_papers,
        })
    return fronts

# ── Render one category section ───────────────────────────────────────────────

def _render_category(category: str, papers: dict, l1: dict, fronts: list) -> str:
    lines = []

    # Sort all papers by date descending
    # kv[1] is either a dict (DB) or pipe-string (JSON fallback)
    def _sort_date(kv):
        v = kv[1]
        if isinstance(v, dict):
            return v.get('date', '0000-00-00')
        return _date_key(str(v))

    sorted_papers = sorted(papers.items(), key=_sort_date, reverse=True)

    lines.append(f"## {category}\n")

    # ── Most Recent (top N) ───────────────────────────────────────────────────
    lines.append("### 🆕 Most Recent\n")
    lines.append("| Date | Title | Authors | Affiliation | Links |")
    lines.append("|------|-------|---------|-------------|-------|")
    shown = 0
    for _, content in sorted_papers:
        if shown >= TOP_N:
            break
        date, title, authors, json_affil, json_venue, aid, json_code = _parse(content)
        info   = l1.get(aid, {})
        if not _is_visible(info):
            continue
        affil  = info.get("affiliations") or json_affil
        brief  = info.get("brief", "")
        links  = _links_cell(aid, json_code, info.get("code_url", ""))
        lines.append(f"| {date} | {_title_cell(title, brief)} | {authors} | {affil} | {links} |")
        shown += 1
    lines.append("")

    # ── Best Papers (top N by M+I+P, L1-analyzed papers only) ────────────────
    lines.append("### ⭐ Best Papers\n")
    scored = []
    for _, content in sorted_papers:
        date, title, authors, json_affil, json_venue, aid, json_code = _parse(content)
        if aid in l1 and _is_visible(l1[aid]):
            scored.append((date, title, authors, json_affil, json_venue, aid, json_code, l1[aid]))
    scored.sort(key=lambda x: x[7]["score"], reverse=True)

    if scored:
        lines.append("| Score | Date | Title | Authors | Affiliation | Links |")
        lines.append("|-------|------|-------|---------|-------------|-------|")
        for date, title, authors, json_affil, json_venue, aid, json_code, info in scored[:TOP_N]:
            affil = info.get("affiliations") or json_affil
            links = _links_cell(aid, json_code, info.get("code_url", ""))
            lines.append(
                f"| {info['score']}/30 | {date} | {_title_cell(title, info['brief'])} "
                f"| {authors} | {affil} | {links} |"
            )
    else:
        lines.append("*No papers with L1 analysis yet.*")
    lines.append("")

    # ── Research Fronts (only when L2 data exists) ────────────────────────────
    if fronts:
        lines.append("### 🔬 Research Fronts\n")
        lines.append("| Status | Front Name | Papers | Key Methods | Problems |")
        lines.append("|--------|-----------|--------|-------------|----------|")
        for f in fronts:
            methods_str  = ", ".join(_readable_tag(m) for m in f["methods"])  if f["methods"]  else "—"
            problems_str = ", ".join(_readable_tag(p) for p in f["problems"]) if f["problems"] else "—"
            emoji = STATUS_EMOJI.get(f["status"], "")
            status_cell = f"{emoji} {f['status'].capitalize()}" if emoji else f['status'].capitalize()

            # Build collapsible paper list inside the Front Name cell
            paper_links = []
            for pid in f.get("core_papers", []):
                clean_pid = _clean_id(pid)
                content = papers.get(clean_pid)
                if content:
                    _, ptitle, _, _, _, _, _ = _parse(content)
                    plain = _plain_title(ptitle)
                    paper_links.append(
                        f'• <a href="http://arxiv.org/abs/{clean_pid}">{plain}</a>'
                    )
            if paper_links:
                inner = "<br>".join(paper_links)
                name_cell = (f"<details><summary>{f['name']}</summary>"
                             f"{inner}</details>")
            else:
                name_cell = f['name']

            lines.append(
                f"| {status_cell} | {name_cell} | {f['size']} "
                f"| {methods_str} | {problems_str} |"
            )
        lines.append("")

    # ── Full list (collapsible HTML table) ────────────────────────────────────
    # Column widths: title is 1.8× the base unit.
    # Base ≈ 9 → title ≈ 16.  Total = 6+8+16+12+14+9+13 = 78 → normalised below.
    # Count visible papers (exclude is_relevant=0 analyzed papers).
    visible_total = sum(
        1 for _, c in sorted_papers
        if _is_visible(l1.get(_parse(c)[5], {}))
    )
    lines.append(f"<details>")
    lines.append(f"<summary>📋 Full list ({visible_total} papers, sorted by date)</summary>\n")
    lines.append(
        '<table><colgroup>'
        '<col width="5%"><col width="7%"><col width="34%">'
        '<col width="13%"><col width="16%"><col width="5%"><col width="20%">'
        '</colgroup>'
    )
    lines.append(
        '<thead><tr>'
        '<th>Score</th><th>Date</th><th>Title</th>'
        '<th>Authors</th><th>Affiliation</th><th>Venue</th><th>Links</th>'
        '</tr></thead>'
    )
    lines.append('<tbody>')
    shown = 0
    for _, content in sorted_papers:
        if shown >= MAX_FULL_LIST:
            break
        date, title, authors, json_affil, json_venue, aid, json_code = _parse(content)
        info      = l1.get(aid, {})
        if not _is_visible(info):
            continue
        affil     = info.get("affiliations") or json_affil
        venue     = info.get("venue") or json_venue
        brief     = info.get("brief", "")
        score_str = f"{info['score']}/30" if info else "—"
        links     = _links_cell_html(aid, json_code, info.get("code_url", ""))
        lines.append(
            f'<tr>'
            f'<td><small>{score_str}</small></td>'
            f'<td><small>{date}</small></td>'
            f'<td>{_title_cell_html(title, brief)}</td>'
            f'<td><small>{authors}</small></td>'
            f'<td><small>{affil}</small></td>'
            f'<td><small>{venue}</small></td>'
            f'<td><small>{links}</small></td>'
            f'</tr>'
        )
        shown += 1
    lines.append('</tbody></table>')
    lines.append("\n</details>\n")

    return "\n".join(lines)

# ── Main generation ───────────────────────────────────────────────────────────

def _load_paper_data(db: Database, json_fallback: Path) -> dict:
    """Load paper list as {category: {arxiv_id: paper_dict_or_pipe_string}}.

    Primary source: `papers` DB table.
    Fallback: JSON file (migrates data into DB on first use).
    """
    row = db.fetchone("SELECT COUNT(*) as cnt FROM papers")
    if row and row['cnt'] > 0:
        by_cat = db.get_all_papers()
        return {cat: {p['arxiv_id']: p for p in plist} for cat, plist in by_cat.items()}

    # papers table is empty — try JSON and migrate
    if json_fallback.exists():
        print(f"  [MIGRATE] papers table empty, seeding from {json_fallback.name} ...", flush=True)
        n = db.migrate_json_to_papers(json_fallback)
        print(f"  [MIGRATE] {n} papers written to papers table", flush=True)
        by_cat = db.get_all_papers()
        if by_cat:
            return {cat: {p['arxiv_id']: p for p in plist} for cat, plist in by_cat.items()}
        # If still empty (unlikely), fall through to raw JSON
        with open(json_fallback, encoding="utf-8") as f:
            return json.load(f)

    print(f"  [WARN] No papers in DB table and JSON not found: {json_fallback}", flush=True)
    return {}


def generate_readme(json_path: Path, out_path: Path, to_web: bool = False):
    """Read papers from DB (papers table) + analysis DB, write enriched markdown to out_path."""
    today = datetime.date.today().isoformat().replace("-", ".")
    sections = []

    with Database() as db:
        data = _load_paper_data(db, json_path)

        if not data:
            print(f"[WARN] No papers found — skipping {out_path.name}", flush=True)
            return

        ordered = sorted(
            data.keys(),
            key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else len(CATEGORY_ORDER)
        )
        for category in ordered:
            papers = data[category]
            if not papers:
                continue
            print(f"  [{category}] {len(papers)} papers ...", flush=True)

            l1 = _load_l1(db, category)

            # Sync affiliations/venue/code: write values to paper_analyses where DB is empty
            for aid, content in papers.items():
                _, _, _, json_affil, json_venue, _, json_code = _parse(content)
                info = l1.get(aid, {})
                needs_affil = json_affil and not info.get("affiliations")
                needs_venue = json_venue and not info.get("venue")
                needs_code  = json_code and json_code.lower() not in ("null", "none", "") \
                              and not info.get("code_url")
                if needs_affil or needs_venue or needs_code:
                    # json_code may be a raw URL (from DB papers table) or **[link](url)** (JSON)
                    if needs_code:
                        url_match = re.search(r'\[link\]\((.+?)\)', json_code)
                        code_url = url_match.group(1) if url_match else (
                            json_code if json_code.startswith("http") else None
                        )
                    else:
                        code_url = None
                    db.update_paper_metadata(
                        aid,
                        affiliations=json_affil if needs_affil else None,
                        venue=json_venue if needs_venue else None,
                        code_url=code_url,
                    )

            # Re-load l1 after sync so affiliations are fresh
            l1 = _load_l1(db, category)

            fronts = _load_fronts(db, category)
            sections.append(_render_category(category, papers, l1, fronts))

    # Table of contents (same order as sections)
    toc_lines = ["<details>", "  <summary>Table of Contents</summary>", "  <ol>"]
    for category in ordered:
        slug = category.replace(" ", "-").lower()
        toc_lines.append(f"    <li><a href=#{slug}>{category}</a></li>")
    toc_lines += ["  </ol>", "</details>\n"]
    toc = "\n".join(toc_lines)

    header = f"## Updated on {today}\n"
    if to_web:
        header = "---\nlayout: default\n---\n\n" + header

    body = header + toc + "\n".join(sections)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"  Written: {out_path}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("GENERATE ENRICHED README", flush=True)
    print("=" * 60, flush=True)

    print(f"\nGenerating README.md ...", flush=True)
    generate_readme(JSON_README, README_PATH, to_web=False)

    print(f"\nGenerating docs/index.md ...", flush=True)
    generate_readme(JSON_GITPAGE, INDEX_PATH, to_web=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
