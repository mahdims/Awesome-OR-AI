#!/usr/bin/env python3
"""
Paper Lookup — inspect all DB fields for a single paper.

Usage:
    python src/scripts/paper_lookup.py https://arxiv.org/abs/2502.04573
    python src/scripts/paper_lookup.py 2502.04573
    python src/scripts/paper_lookup.py 2502.04573 --json
    python src/scripts/paper_lookup.py 2502.04573 --section methodology
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 output on Windows (avoids UnicodeEncodeError with box chars / arrows)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from db.database import Database

# ── Display constants ─────────────────────────────────────────────────────────

HEAVY = "=" * 68
LIGHT = "-" * 68
WRAP_WIDTH = 74
INDENT = "  "

SECTIONS = [
    "metadata", "brief", "relevance", "problem",
    "methodology", "experiments", "results", "artifacts",
    "tags", "lineage", "extensions", "abstract",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json(value, default=None):
    """Parse a JSON string stored in the DB, or return default."""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _wrap(text, width=WRAP_WIDTH, indent=INDENT):
    """Word-wrap text with a leading indent."""
    if not text:
        return indent + "(none)"
    lines = []
    for para in str(text).split("\n"):
        if para.strip():
            lines.append(textwrap.fill(para, width=width,
                                       initial_indent=indent,
                                       subsequent_indent=indent))
        else:
            lines.append("")
    return "\n".join(lines)


def _list(items, sep=" / ", fallback="(none)"):
    """Join a list into a readable string."""
    if not items:
        return fallback
    return sep.join(str(i) for i in items)


def _yn(val):
    """Boolean / integer to YES / NO."""
    return "YES" if val else "NO"


def _field(label, value, width=18):
    """Left-aligned label : value line."""
    return f"{INDENT}{label:<{width}}: {value}"


def _section_header(title):
    print(f"\n{LIGHT}")
    print(title)


def extract_arxiv_id(arg: str) -> str:
    """Extract bare arxiv ID from a URL or plain string."""
    m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", arg)
    return m.group(1) if m else arg.strip()


# ── Section printers ─────────────────────────────────────────────────────────

def print_metadata(p: dict, row: dict):
    print("\nMETADATA")
    rel = _json(row.get("relevance"), {})
    sig = _json(row.get("significance"), {})
    max_mpi = max(rel.get("methodological", 0),
                  rel.get("problem", 0),
                  rel.get("inspirational", 0))
    is_rel = row.get("is_relevant", 1)
    rel_label = f"{'YES' if is_rel else 'NO'}  (max MPI = {max_mpi})"

    authors = _json(row.get("authors"), [])
    print(_field("Title",        row.get("title", "(unknown)")))
    print(_field("Authors",      _list(authors, sep=", ")))
    print(_field("Published",    row.get("published_date", "?")))
    print(_field("Category",     row.get("category", "?")))
    print(_field("Affiliations", row.get("affiliations") or "(none)"))
    print(_field("ArXiv URL",    f"https://arxiv.org/abs/{row['arxiv_id']}"))
    model = row.get("analysis_model") or "?"
    date  = (row.get("analysis_date") or "?")[:10]
    print(_field("Analyzed",     f"{date}  (model: {model})"))
    print(_field("is_relevant",  rel_label))


def print_brief(p: dict, row: dict):
    _section_header("BRIEF  (Agent 3 — Positioning)")
    print(_wrap(row.get("brief", "")))


def print_relevance(p: dict, row: dict):
    _section_header("RELEVANCE SCORES  (Agent 3)")
    rel = _json(row.get("relevance"), {})
    print(_field("Methodological", f"{rel.get('methodological', '?')} / 10"))
    print(_field("Problem",        f"{rel.get('problem', '?')} / 10"))
    print(_field("Inspirational",  f"{rel.get('inspirational', '?')} / 10"))

    print("\nSIGNIFICANCE")
    sig = _json(row.get("significance"), {})
    print(_field("Must-Read",        _yn(sig.get("must_read"))))
    print(_field("Changes Thinking", _yn(sig.get("changes_thinking"))))
    print(_field("Team Discussion",  _yn(sig.get("team_discussion"))))
    reasoning = sig.get("reasoning", "")
    if reasoning:
        print(f"{INDENT}Reasoning :")
        print(_wrap(reasoning, indent=INDENT + "  "))


def print_problem(p: dict, row: dict):
    _section_header("PROBLEM  (Agent 1 — Reader)")
    prob = _json(row.get("problem"), {})
    print(_field("Formal name", prob.get("formal_name", "?")))
    print(_field("Short",       prob.get("short", "?")))
    print(_field("Class",       prob.get("class_", prob.get("class", "?"))))
    print(_field("Properties",  _list(prob.get("properties", []))))
    print(_field("Scale",       prob.get("scale", "?")))


def print_methodology(p: dict, row: dict):
    _section_header("METHODOLOGY  (Agent 1 — Reader)")
    meth = _json(row.get("methodology"), {})
    print(_field("Core method",       meth.get("core_method", "?")))
    print(_field("LLM role",          meth.get("llm_role", "?")))
    print(_field("LLM model used",    meth.get("llm_model_used") or "(none)"))
    print(_field("Search type",       meth.get("search_type", "?")))
    print(_field("Training required", _yn(meth.get("training_required"))))
    novelty = meth.get("novelty_claim", "")
    if novelty:
        print(f"{INDENT}Novelty claim :")
        print(_wrap(novelty, indent=INDENT + "  "))
    print(_field("Components", _list(meth.get("components", []))))


def print_experiments(p: dict, row: dict):
    _section_header("EXPERIMENTS  (Agent 1)")
    exp = _json(row.get("experiments"), {})
    sizes = exp.get("instance_sizes", [])
    print(_field("Benchmarks",    _list(exp.get("benchmarks", []))))
    print(_field("Baselines",     _list(exp.get("baselines", []))))
    print(_field("Hardware",      exp.get("hardware") or "(none)"))
    print(_field("Instance sizes", _list([str(s) for s in sizes])))


def print_results(p: dict, row: dict):
    print("\nRESULTS")
    res = _json(row.get("results"), {})
    vs = res.get("vs_baselines", {})
    if vs:
        vs_str = ",  ".join(f"{k} → {v}" for k, v in vs.items())
        print(f"{INDENT}vs baselines :")
        print(_wrap(vs_str, indent=INDENT + "  "))
    else:
        print(_field("vs baselines", "(none)"))
    print(_field("Scalability", res.get("scalability") or "?"))
    print(_field("Statistical",  res.get("statistical_rigor") or "?"))
    print(_field("Limitations",  _list(res.get("limitations_acknowledged", []))))


def print_artifacts(p: dict, row: dict):
    _section_header("ARTIFACTS")
    art = _json(row.get("artifacts"), {})
    print(_field("Code URL",        art.get("code_url") or "(none)"))
    print(_field("Models released", _yn(art.get("models_released"))))
    print(_field("New benchmark",   _yn(art.get("new_benchmark"))))


def print_tags(p: dict, row: dict):
    _section_header("TAGS  (Agent 2 — Methods)")
    tags = _json(row.get("tags"), {})
    print(_field("Methods",           _list(tags.get("methods", []))))
    print(_field("Problems",          _list(tags.get("problems", []))))
    print(_field("Contribution type", _list(tags.get("contribution_type", []))))
    print(_field("Framework lineage", tags.get("framework_lineage") or "(none)"))
    print(_field("Specific domain",   tags.get("specific_domain") or "(none)"))
    print(_field("LLM coupling",      tags.get("llm_coupling") or "(none)"))

    print("\nLINEAGE")
    lin = _json(row.get("lineage"), {})
    print(_field("Novelty type",  lin.get("novelty_type", "?")))
    closest = lin.get("closest_prior_work", "")
    if closest:
        print(f"{INDENT}Closest prior :")
        print(_wrap(closest, indent=INDENT + "  "))
    ancestors = lin.get("direct_ancestors", [])
    if ancestors:
        print(f"{INDENT}Direct ancestors :")
        for i, anc in enumerate(ancestors, 1):
            paper = anc.get("paper", "?")
            rel   = anc.get("relationship", "")
            print(f"{INDENT}  {i}. {paper}")
            if rel:
                print(f"{INDENT}     — {rel}")

    print("\nEXTENSIONS")
    ext = _json(row.get("extensions"), {})
    print(_field("Next steps",      _list(ext.get("next_steps", [])[:3])))
    print(_field("Transferable to", _list(ext.get("transferable_to", [])[:3])))
    print(_field("Open weaknesses", _list(ext.get("open_weaknesses", [])[:3])))


def print_abstract(p: dict, row: dict):
    _section_header("ABSTRACT")
    print(_wrap(row.get("abstract", "(no abstract stored)")))


# Map section name → printer function
_PRINTERS = {
    "metadata":    print_metadata,
    "brief":       print_brief,
    "relevance":   print_relevance,
    "problem":     print_problem,
    "methodology": print_methodology,
    "experiments": print_experiments,
    "results":     print_results,
    "artifacts":   print_artifacts,
    "tags":        print_tags,
    "lineage":     print_tags,       # lineage is printed inside tags block
    "extensions":  print_tags,       # extensions too
    "abstract":    print_abstract,
}

# Ordered full display — experiments/results share a printer so dedup
_FULL_ORDER = [
    "metadata", "brief", "relevance", "problem",
    "methodology", "experiments", "artifacts", "tags", "abstract",
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Look up all DB information for a single paper."
    )
    parser.add_argument(
        "arxiv",
        help="ArXiv URL (https://arxiv.org/abs/XXXX.XXXXX) or bare ID (XXXX.XXXXX)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump the raw DB row as JSON instead of pretty-printing"
    )
    parser.add_argument(
        "--section",
        choices=SECTIONS,
        metavar="SECTION",
        help=f"Print only one section. Choices: {', '.join(SECTIONS)}"
    )
    args = parser.parse_args()

    arxiv_id = extract_arxiv_id(args.arxiv)

    db = Database()
    with db:
        row = db.get_analysis(arxiv_id)

    if row is None:
        print(f"\n[ERROR] Paper '{arxiv_id}' not found in database.")
        print(f"        Run:  python src/scripts/layer1_analyze_new.py")
        print(f"        to analyze new papers first.\n")
        return 1

    # ── JSON mode ────────────────────────────────────────────────────────────
    if args.json:
        print(json.dumps(row, indent=2, default=str))
        return 0

    # ── Pretty-print ─────────────────────────────────────────────────────────
    print(f"\n{HEAVY}")
    print(f"PAPER: {arxiv_id}")
    print(HEAVY)

    p = {}  # enriched placeholder (unused — printers read raw row directly)

    if args.section:
        # Single section
        sec = args.section
        # experiments and results share print_experiments / print_results
        if sec == "experiments":
            print_experiments(p, row)
            print_results(p, row)
        elif sec in ("lineage", "extensions"):
            print_tags(p, row)  # prints tags + lineage + extensions together
        else:
            _PRINTERS[sec](p, row)
    else:
        # Full output — experiments followed by results inline
        for sec in _FULL_ORDER:
            _PRINTERS[sec](p, row)
            if sec == "experiments":
                print_results(p, row)

    print(f"\n{HEAVY}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
