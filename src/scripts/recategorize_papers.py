"""
Interactive recategorization tool for "OR for Generative AI" papers.

Uses an LLM to classify each paper (title + abstract from DB) into the best-fit category.
Papers the LLM wants to keep are silently skipped.
Papers the LLM wants to move trigger an interactive prompt where the user
can accept the suggestion (Enter) or override (1/2/3).

Changes are applied to both JSON files and the SQLite database.

Usage:
    python src/scripts/recategorize_papers.py [--config_path config.yaml] [--dry-run]
"""

import os
import sys
import re
import json
import time
import argparse

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.layer0.fetch_and_score import load_config

try:
    from db.database import Database
except ImportError:
    Database = None

try:
    from google import genai
except ImportError:
    genai = None

SOURCE_CATEGORY = "OR for Generative AI"
CATEGORIES = [
    "LLMs for Algorithm Design",
    "Generative AI for OR",
    "OR for Generative AI",
]

CLASSIFY_PROMPT = """You are a research paper classifier. Classify the paper into exactly one of three categories.

== DECISION RULE ==
Ask yourself: what is the TOOL and what is the SUBJECT?
- If the TOOL is an LLM/AI and the SUBJECT is an OR problem (routing, scheduling, combinatorial optimization, MIP modeling) → Category 2
- If the TOOL is an LLM/AI and the SUBJECT is designing/evolving algorithms or heuristics → Category 1
- If the TOOL is a formal OR method (LP, MIP, queueing, scheduling theory, game theory with optimization) and the SUBJECT is an AI system (LLM behavior, safety, alignment, inference, serving, multi-agent AI, GPU allocation) → Category 3

== CATEGORIES ==
1. LLMs for Algorithm Design
   The paper uses LLMs/AI to automatically design, evolve, or synthesize algorithms and heuristics.
   Examples: AlphaEvolve-style evolutionary search, FunSearch, automated heuristic design, LLM-guided metaheuristic discovery, program synthesis for optimization algorithms.

2. Generative AI for OR
   The paper uses LLMs/AI as a tool to model or solve a classical operations research problem.
   Examples: LLM formulates a MIP for a logistics problem, NL-to-optimization pipeline, AI agent solves vehicle routing or scheduling, LLM generates constraint programs.

3. OR for Generative AI
   The paper uses formal OR/mathematical optimization methods as a tool to improve, control, or operate an AI system.
   The AI system is the SUBJECT being optimized — not the solver.
   Examples: LP/MIP for GPU scheduling, queueing theory for LLM inference, game-theoretic optimization for LLM safety/alignment, MIP for LLM serving resource allocation, scheduling theory for multi-agent AI coordination, linear programming applied to LLM behavior or policy.

== IMPORTANT ==
Category 3 is NOT limited to infrastructure. ANY paper that uses an OR method (LP, MIP, game theory, queueing, scheduling) to optimize or control an AI/LLM system — including safety, alignment, dialogue policy, or agent behavior — belongs here.

Paper title: "{title}"
Abstract: {abstract}

Respond with ONLY a JSON object: {{"category": "<category name>", "reason": "<one sentence>"}}
"""


def _extract_title(entry: str, paper_id: str) -> str:
    """Extract title from a pipe-delimited or markdown paper entry."""
    # Try pipe-split first: |date|title|authors|...
    parts = [p.strip() for p in entry.split("|")]
    parts = [p for p in parts if p]  # remove empty
    if len(parts) >= 2:
        # Strip markdown formatting from second field (title)
        title = re.sub(r'\*\*(.+?)\*\*', r'\1', parts[1])
        if len(title) > 10:
            return title
    # Fallback: grab second **bold** item
    matches = re.findall(r'\*\*(.+?)\*\*', entry)
    return matches[1] if len(matches) > 1 else paper_id


def _get_abstract_from_db(db, paper_id: str) -> str:
    """Try to fetch abstract from paper_analyses or rescore_cache tables."""
    if db is None:
        return ""
    row = db.fetchone(
        "SELECT abstract FROM paper_analyses WHERE arxiv_id = ?", (paper_id,)
    )
    if row and row["abstract"]:
        return row["abstract"]
    # Fallback: rescore_cache (populated by rescore_and_filter.py)
    try:
        row = db.fetchone(
            "SELECT abstract FROM rescore_cache WHERE arxiv_id = ? AND abstract != ''",
            (paper_id,),
        )
        return (row["abstract"] or "") if row else ""
    except Exception:
        return ""


def _classify_title(client, title: str, abstract: str = "") -> tuple[str, str]:
    """Ask LLM to classify a paper by title+abstract. Returns (category, reason)."""
    abstract_text = abstract.strip() if abstract else "(not available)"
    prompt = CLASSIFY_PROMPT.format(title=title, abstract=abstract_text)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        category = result.get("category", SOURCE_CATEGORY)
        reason = result.get("reason", "")
        # Validate category
        if category not in CATEGORIES:
            return SOURCE_CATEGORY, reason
        return category, reason
    except Exception as e:
        print(f"    (LLM error: {e})", flush=True)
        return SOURCE_CATEGORY, "classification failed"


def _move_paper_in_json(data: dict, paper_id: str, entry: str,
                        from_cat: str, to_cat: str) -> None:
    """Move paper_id from from_cat to to_cat within the data dict."""
    if from_cat in data and paper_id in data[from_cat]:
        del data[from_cat][paper_id]
    if to_cat not in data:
        data[to_cat] = {}
    data[to_cat][paper_id] = entry


def _move_paper_in_db(db, paper_id: str, from_cat: str, to_cat: str) -> None:
    """Move paper in SQLite: copy row to new category, delete old row."""
    if db is None:
        return

    # Move in `papers` table
    row = db.fetchone(
        "SELECT * FROM papers WHERE arxiv_id = ? AND category = ?",
        (paper_id, from_cat),
    )
    if row:
        db.execute(
            """INSERT OR REPLACE INTO papers
               (arxiv_id, category, title, authors, date, affiliation, venue, code_url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (paper_id, to_cat, row["title"], row["authors"], row["date"],
             row["affiliation"], row["venue"], row["code_url"], row["fetched_at"]),
        )
        db.execute(
            "DELETE FROM papers WHERE arxiv_id = ? AND category = ?",
            (paper_id, from_cat),
        )

    # Move in `paper_analyses` table if present
    row_a = db.fetchone(
        "SELECT * FROM paper_analyses WHERE arxiv_id = ? AND category = ?",
        (paper_id, from_cat),
    )
    if row_a:
        # Build column list dynamically
        cols = list(row_a.keys())
        vals = [row_a[c] for c in cols]
        cat_idx = cols.index("category")
        vals[cat_idx] = to_cat
        placeholders = ", ".join("?" * len(cols))
        col_names = ", ".join(cols)
        db.execute(
            f"INSERT OR REPLACE INTO paper_analyses ({col_names}) VALUES ({placeholders})",
            tuple(vals),
        )
        db.execute(
            "DELETE FROM paper_analyses WHERE arxiv_id = ? AND category = ?",
            (paper_id, from_cat),
        )

    db.commit()


def recategorize(config, dry_run: bool = False) -> None:
    print("=" * 60, flush=True)
    print("recategorize_papers — interactive OR-for-GenAI cleanup", flush=True)
    print("=" * 60, flush=True)

    # Init Gemini client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        print("ERROR: GEMINI_API_KEY not set or google-genai not installed.", flush=True)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    # Init DB
    db = None
    if Database is not None:
        db = Database()
        db.connect()
        print(f"Database: {db.db_path}", flush=True)

    # Collect JSON paths
    json_paths = []
    if config.get("publish_readme"):
        json_paths.append(config["json_readme_path"])
    if config.get("publish_gitpage"):
        json_paths.append(config["json_gitpage_path"])

    if not json_paths:
        print("ERROR: No JSON output path configured.", flush=True)
        sys.exit(1)

    # Load ALL json files into memory upfront so we can write-on-confirm
    json_data: dict[str, dict] = {}
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data[json_path] = json.loads(f.read())

    papers_in_scope = dict(json_data[json_paths[0]].get(SOURCE_CATEGORY, {}))
    total = len(papers_in_scope)
    print(f"\nPapers in '{SOURCE_CATEGORY}': {total}", flush=True)
    if total == 0:
        print("Nothing to recategorize.", flush=True)
        if db:
            db.close()
        return

    moved = 0
    kept = 0

    for i, (paper_id, entry) in enumerate(papers_in_scope.items(), 1):
        title = _extract_title(str(entry), paper_id)
        abstract = _get_abstract_from_db(db, paper_id)
        suggested_cat, reason = _classify_title(client, title, abstract)
        time.sleep(0.2)  # gentle rate limiting

        if suggested_cat == SOURCE_CATEGORY:
            kept += 1
            continue  # silently keep

        # LLM wants to move — prompt user
        print(f"\n[{i}/{total}] {title[:80]}")
        print(f"  LLM suggests → {suggested_cat}  ({reason})")
        print(f"  [Enter] Accept   [1] {CATEGORIES[0]}   [2] {CATEGORIES[1]}   [3] {CATEGORIES[2]}")
        raw = input("  > ").strip()

        if raw == "":
            target = suggested_cat
        elif raw in ("1", "2", "3"):
            target = CATEGORIES[int(raw) - 1]
        else:
            print("  (skipped — paper stays in OR for Generative AI)", flush=True)
            kept += 1
            continue

        if target == SOURCE_CATEGORY:
            print(f"  ✓ Keeping in '{SOURCE_CATEGORY}'", flush=True)
            kept += 1
            continue

        # Apply immediately to all JSON files in memory and write to disk
        if not dry_run:
            for json_path, data in json_data.items():
                e = data.get(SOURCE_CATEGORY, {}).get(paper_id)
                if e is not None:
                    _move_paper_in_json(data, paper_id, e, SOURCE_CATEGORY, target)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=None)
            # Apply immediately to DB
            _move_paper_in_db(db, paper_id, SOURCE_CATEGORY, target)
            print(f"  ✓ Moved to '{target}' — JSON + DB updated", flush=True)
        else:
            print(f"  [DRY RUN] Would move to '{target}'", flush=True)
        moved += 1

    if db:
        db.close()

    print(f"\nDone. Moved: {moved}, Kept: {kept}, "
          f"Auto-kept (LLM agreed): {total - moved - kept}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively recategorize OR-for-GenAI papers")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show proposed moves without applying them")
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    recategorize(cfg, dry_run=args.dry_run)
