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

# ── Status emoji mapping (L2 research front status) ──────────────────────────

STATUS_EMOJI = {
    "emerging":  "🚀",
    "growing":   "📈",
    "stable":    "✅",
    "declining": "📉",
    "merged":    "🔀",
    "split":     "🔀",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_id(arxiv_id: str) -> str:
    return re.sub(r'v\d+$', '', arxiv_id.strip())


def _parse(s: str):
    """Return (date, title, authors, affiliation, venue, arxiv_id, code)."""
    parts = s.split("|")
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


def _code_cell(code: str) -> str:
    code = code.strip()
    if not code or code.lower() in ("null", "none"):
        return "—"
    return code  # already formatted as **[link](url)**


def _pdf_cell(arxiv_id: str) -> str:
    return f"[{arxiv_id}](http://arxiv.org/abs/{arxiv_id})"

# ── DB queries ────────────────────────────────────────────────────────────────

def _load_l1(db: Database, category: str) -> dict:
    """Return {arxiv_id: {score, brief, affiliations, venue, code_url}} for all analyzed papers."""
    rows = db.fetchall(
        """SELECT arxiv_id, relevance, significance, brief, affiliations, venue, artifacts
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
        """SELECT name, status, size, dominant_methods, dominant_problems
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
        fronts.append({
            "name":     r["name"] or "Unnamed front",
            "status":   r["status"] or "stable",
            "size":     r["size"] or 0,
            "methods":  methods,
            "problems": problems,
        })
    return fronts

# ── Render one category section ───────────────────────────────────────────────

def _render_category(category: str, papers: dict, l1: dict, fronts: list) -> str:
    lines = []

    # Sort all papers by date descending
    sorted_papers = sorted(
        papers.items(),
        key=lambda kv: _date_key(str(kv[1])),
        reverse=True,
    )

    lines.append(f"## {category}\n")

    # ── Most Recent (top N) ───────────────────────────────────────────────────
    lines.append("### 🆕 Most Recent\n")
    lines.append("| Date | Title | Authors | Affiliation | Code |")
    lines.append("|------|-------|---------|-------------|------|")
    for _, content in sorted_papers[:TOP_N]:
        date, title, authors, json_affil, json_venue, aid, json_code = _parse(str(content))
        info   = l1.get(aid, {})
        affil  = info.get("affiliations") or json_affil
        db_code = info.get("code_url")
        code_cell = _code_cell(f"**[link]({db_code})**" if db_code else json_code)
        brief  = info.get("brief", "")
        lines.append(f"| {date} | {_title_cell(title, brief)} | {authors} | {affil} | {code_cell} |")
    lines.append("")

    # ── Best Papers (top N by M+I+P, L1-analyzed papers only) ────────────────
    lines.append("### ⭐ Best Papers\n")
    scored = []
    for _, content in sorted_papers:
        date, title, authors, json_affil, json_venue, aid, json_code = _parse(str(content))
        if aid in l1:
            scored.append((date, title, authors, json_affil, json_venue, aid, json_code, l1[aid]))
    scored.sort(key=lambda x: x[7]["score"], reverse=True)

    if scored:
        lines.append("| Score | Date | Title | Authors | Affiliation | Code |")
        lines.append("|-------|------|-------|---------|-------------|------|")
        for date, title, authors, json_affil, json_venue, aid, json_code, info in scored[:TOP_N]:
            affil = info.get("affiliations") or json_affil
            db_code = info.get("code_url")
            code_cell = _code_cell(f"**[link]({db_code})**" if db_code else json_code)
            lines.append(
                f"| {info['score']}/30 | {date} | {_title_cell(title, info['brief'])} "
                f"| {authors} | {affil} | {code_cell} |"
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
            methods_str  = ", ".join(f["methods"])  if f["methods"]  else "—"
            problems_str = ", ".join(f["problems"]) if f["problems"] else "—"
            lines.append(
                f"| {f['status'].capitalize()} "
                f"| {f['name']} | {f['size']} "
                f"| {methods_str} | {problems_str} |"
            )
        lines.append("")

    # ── Full list (collapsible) ───────────────────────────────────────────────
    total = len(sorted_papers)
    lines.append(f"<details>")
    lines.append(f"<summary>📋 Full list ({total} papers, sorted by date)</summary>\n")
    lines.append("| Score | Date | Title | Authors | Affiliation | Venue | PDF | Code |")
    lines.append("|-------|------|-------|---------|-------------|-------|-----|------|")
    for _, content in sorted_papers[:MAX_FULL_LIST]:
        date, title, authors, json_affil, json_venue, aid, json_code = _parse(str(content))
        info      = l1.get(aid, {})
        affil     = info.get("affiliations") or json_affil
        venue     = info.get("venue") or json_venue
        brief     = info.get("brief", "")
        score_str = f"{info['score']}/30" if info else "—"
        db_code   = info.get("code_url")
        code_cell = _code_cell(f"**[link]({db_code})**" if db_code else json_code)
        lines.append(
            f"| {score_str} | {date} | {_title_cell(title, brief)} "
            f"| {authors} | {affil} | {venue} | {_pdf_cell(aid)} | {code_cell} |"
        )
    lines.append("\n</details>\n")

    return "\n".join(lines)

# ── Main generation ───────────────────────────────────────────────────────────

def generate_readme(json_path: Path, out_path: Path, to_web: bool = False):
    """Read JSON + DB and write enriched markdown to out_path."""
    if not json_path.exists():
        print(f"[WARN] JSON not found: {json_path}", flush=True)
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    today = datetime.date.today().isoformat().replace("-", ".")
    sections = []

    with Database() as db:
        for category, papers in data.items():
            if not papers:
                continue
            print(f"  [{category}] {len(papers)} papers ...", flush=True)

            l1 = _load_l1(db, category)

            # Sync affiliations/venue/code: write JSON values to DB where DB is empty
            for _, content in papers.items():
                _, _, _, json_affil, json_venue, aid, json_code = _parse(str(content))
                info = l1.get(aid, {})
                needs_affil = json_affil and not info.get("affiliations")
                needs_venue = json_venue and not info.get("venue")
                needs_code  = json_code and json_code.lower() not in ("null", "none", "") \
                              and not info.get("code_url")
                if needs_affil or needs_venue or needs_code:
                    url_match = re.search(r'\[link\]\((.+?)\)', json_code) if needs_code else None
                    db.update_paper_metadata(
                        aid,
                        affiliations=json_affil if needs_affil else None,
                        venue=json_venue if needs_venue else None,
                        code_url=url_match.group(1) if url_match else None,
                    )

            # Re-load l1 after sync so affiliations are fresh
            l1 = _load_l1(db, category)

            fronts = _load_fronts(db, category)
            sections.append(_render_category(category, papers, l1, fronts))

    # Table of contents
    toc_lines = ["<details>", "  <summary>Table of Contents</summary>", "  <ol>"]
    for category in data:
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
