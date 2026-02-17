"""
Layer 2: Front Summarizer

Uses LLM to generate three structured outputs per research front in a single call:
  - name            : 6-10 word human-readable title
  - summary         : 3-paragraph narrative (theme / contributions / trajectory)
  - future_directions: list of 3-5 concrete next research steps

Uses all available Layer 1 fields per paper for richer context, including
experiments, results, lineage, extensions, and significance.
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel
from db.database import Database
from llm_client import create_agent_client


class FrontEnrichment(BaseModel):
    """Structured output schema for front LLM enrichment."""
    name: str
    summary: str
    future_directions: List[str]

SYSTEM_PROMPT = """You are a research intelligence analyst specializing in Operations Research and AI.

Given a cluster of related papers (a "research front"), produce a structured JSON response with three fields:

1. **name** (string, 6-10 words): A precise, specific title capturing the front's core theme.
   - Focus on SPECIFIC, DISCRIMINATING aspects (framework names, target domains) that uniquely define THIS front.
   - Avoid generic category-level tags that appear across many fronts (you'll see which tags are common below).
   - Good: "AlphaEvolve-Based Routing Heuristics" (specific framework + domain)
   - Bad: "LLM-Guided Evolutionary Heuristics" (too generic, applies to all fronts in this category)

2. **summary** (string, 3 paragraphs separated by \\n\\n):
   - Paragraph 1: The SPECIFIC unifying theme — use framework names (AlphaEvolve, FunSearch, EoH), target problem domains (combinatorial routing, matrix multiplication), NOT generic tags.
   - Paragraph 2: Key contributions — name specific methods, benchmarks, and quantitative results.
   - Paragraph 3: Trajectory — is this front emerging, maturing, or converging? What's the likely next paper?

3. **future_directions** (list of 3-5 strings): Concrete, actionable research directions.
   - Each item must reference specific methods, problems, or gaps identified in these papers.
   - Bad: "Explore new applications"
   - Good: "Extend AlphaEvolve framework to multi-objective CVRPTW with stochastic demand"

CRITICAL: You will see "COMMON TAGS ACROSS FRONTS" below — these are tags shared by multiple fronts.
DO NOT use these common tags as the main differentiator in the name or summary. Instead, focus on the
SPECIFIC tags (framework_lineage, specific_domain, llm_coupling) that make THIS front unique.

Be specific. Use framework names, target domain names, and quantitative results from the papers.
Return ONLY the JSON object, no preamble."""


def _parse(value, default=None):
    """Safely parse a JSON string or return as-is."""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _join(lst: list, limit: int = 5, sep: str = ", ") -> str:
    if not lst:
        return ""
    return sep.join(str(x) for x in lst[:limit])


def _normalize_future_directions(value) -> List[str]:
    """Normalize future_directions to a clean list of non-empty strings."""
    parsed = _parse(value, [])
    if isinstance(parsed, str):
        parsed = [line.strip() for line in parsed.split('\n') if line.strip()]
    if not isinstance(parsed, list):
        return []
    cleaned = []
    for item in parsed:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _has_valid_enrichment(front: Dict) -> bool:
    """Return True when front already has usable LLM enrichment fields."""
    name = (front.get('name') or '').strip()
    summary = (front.get('summary') or '').strip()
    future_directions = _normalize_future_directions(front.get('future_directions'))
    if name.startswith('[Enrichment failed]'):
        return False
    if summary.startswith('[Summary generation failed'):
        return False
    return bool(name and summary and future_directions)


def build_front_context(front: Dict, db: Database, common_tags: Dict = None) -> str:
    """
    Build rich context string for LLM enrichment of a front.

    Uses 11 Layer 1 fields per paper (up from 6 in the old summarizer):
    title, published_date, problem, methodology (core_method, llm_role, novelty_claim),
    lineage, experiments, results, extensions, significance, brief.

    Args:
        front: Front dict with core_papers, dominant_methods, etc.
        db: Database instance
        common_tags: Optional dict with keys 'methods', 'problems', 'framework_lineage',
                     'specific_domain' — tags that appear in multiple fronts (to avoid focusing on)
    """
    lines = []
    lines.append(f"RESEARCH FRONT: {front['front_id']}")
    lines.append(f"Category: {front['category']}")
    lines.append(f"Size: {front['size']} papers  |  "
                 f"Density: {front.get('internal_density', 0):.3f}  |  "
                 f"Status: {front.get('status', 'unknown')}")

    # Show common tags across fronts (to AVOID focusing on these)
    if common_tags:
        lines.append(f"\n{'─'*60}")
        lines.append("COMMON TAGS ACROSS FRONTS (do NOT use these as main differentiators):")
        for tag_type, tags in common_tags.items():
            if tags:
                lines.append(f"  {tag_type}: {_join(tags, limit=8)}")
        lines.append(f"{'─'*60}")

    methods = _parse(front.get('dominant_methods'), [])
    if methods:
        lines.append(f"Dominant methods : {_join(methods)}")

    problems = _parse(front.get('dominant_problems'), [])
    if problems:
        lines.append(f"Dominant problems: {_join(problems)}")

    lines.append(f"\n{'='*60}")
    lines.append("PAPERS IN THIS FRONT:")
    lines.append(f"{'='*60}")

    papers = _parse(front.get('core_papers'), [])

    for i, paper_id in enumerate(papers, 1):
        a = db.get_analysis(paper_id)
        if not a:
            lines.append(f"\n[{i}] {paper_id}  (no Layer 1 analysis)")
            continue

        pub = a.get('published_date', '')[:7]
        lines.append(f"\n[{i}] {pub} — {paper_id}")
        lines.append(f"  Title     : {a.get('title', '')}")

        # Problem
        prob = _parse(a.get('problem'), {})
        if prob:
            lines.append(f"  Problem   : {prob.get('formal_name', '')} "
                         f"({prob.get('class_', '')})")

        # Methodology
        meth = _parse(a.get('methodology'), {})
        if meth:
            lines.append(f"  Method    : {meth.get('core_method', '')} "
                         f"| LLM role: {meth.get('llm_role', '?')}")
            novelty = (meth.get('novelty_claim') or '')[:100]
            if novelty:
                lines.append(f"  Novelty   : {novelty}")

        # Fine-grained tags (NEW)
        tags = _parse(a.get('tags'), {})
        if tags:
            fw_lin = tags.get('framework_lineage')
            sp_dom = tags.get('specific_domain')
            llm_cpl = tags.get('llm_coupling')
            if fw_lin or sp_dom or llm_cpl:
                parts = []
                if fw_lin:
                    parts.append(f"framework={fw_lin}")
                if sp_dom:
                    parts.append(f"domain={sp_dom}")
                if llm_cpl:
                    parts.append(f"coupling={llm_cpl}")
                lines.append(f"  Tags      : {' | '.join(parts)}")

        # Lineage
        lin = _parse(a.get('lineage'), {})
        if lin:
            closest = (lin.get('closest_prior_work') or '')[:70]
            ntype = lin.get('novelty_type', '')
            if closest or ntype:
                lines.append(f"  Lineage   : {ntype} — closest: {closest}")

        # Experiments
        exp = _parse(a.get('experiments'), {})
        if exp:
            benches = _join(exp.get('benchmarks', []), 4)
            if benches:
                lines.append(f"  Benchmarks: {benches}")

        # Results
        res = _parse(a.get('results'), {})
        if res:
            vs = res.get('vs_baselines', {})
            if isinstance(vs, dict) and vs:
                # Show first 2 comparisons
                items = list(vs.items())[:2]
                vs_str = "; ".join(f"{k}: {v}" for k, v in items)
                lines.append(f"  Results   : {vs_str}")
            limits = _join(res.get('limitations_acknowledged', []), 2)
            if limits:
                lines.append(f"  Limits    : {limits}")

        # Extensions / future work from Layer 1
        ext = _parse(a.get('extensions'), {})
        if ext:
            next_steps = _join(ext.get('next_steps', []), 3, "; ")
            if next_steps:
                lines.append(f"  Next steps: {next_steps}")

        # Significance
        sig = _parse(a.get('significance'), {})
        if sig:
            flags = []
            if sig.get('must_read'):
                flags.append('MUST-READ')
            if sig.get('changes_thinking'):
                flags.append('CHANGES-THINKING')
            reasoning = (sig.get('reasoning') or '')[:80]
            if flags or reasoning:
                lines.append(f"  Signif.   : {' | '.join(flags)}  {reasoning}")

        # Brief (full, not truncated)
        brief = (a.get('brief') or '')
        if brief:
            lines.append(f"  Brief     : {brief}")

    return '\n'.join(lines)


def enrich_front(front: Dict, db: Database, llm_client=None, common_tags: Dict = None) -> Dict:
    """
    Generate name, summary, and future_directions for a research front.

    Makes a single LLM call using generate_json and returns a dict with
    the three fields. Falls back gracefully if the call fails.

    Args:
        front:      Front dict with core_papers, dominant_methods, etc.
        db:         Connected Database instance
        llm_client: Optional pre-created LLM client
        common_tags: Optional dict of tags appearing in multiple fronts (to avoid focusing on)

    Returns:
        Dict with keys: name, summary, future_directions
    """
    if llm_client is None:
        llm_client = create_agent_client('front_summarizer')

    context = build_front_context(front, db, common_tags)

    result = llm_client.generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=context,
        output_schema=FrontEnrichment
    )

    # generate_json returns a Pydantic model instance — convert to dict
    if hasattr(result, 'model_dump'):
        result_dict = result.model_dump()   # Pydantic v2
    elif hasattr(result, 'dict'):
        result_dict = result.dict()         # Pydantic v1
    else:
        result_dict = dict(result)          # fallback

    fd = _normalize_future_directions(result_dict.get('future_directions', []))

    return {
        'name': (result_dict.get('name') or '').strip(),
        'summary': (result_dict.get('summary') or '').strip(),
        'future_directions': fd,
    }


def _compute_common_tags(fronts: List[Dict], db: Database) -> Dict:
    """
    Compute tags that appear in multiple fronts (to tell the LLM to avoid focusing on these).

    Returns dict with keys: 'methods', 'problems', 'framework_lineage', 'specific_domain'
    """
    from collections import Counter

    methods_per_front = Counter()
    problems_per_front = Counter()
    lineage_per_front = Counter()
    domain_per_front = Counter()

    for front in fronts:
        papers = _parse(front.get('core_papers'), [])
        front_methods = set()
        front_problems = set()
        front_lineage = set()
        front_domain = set()

        for paper_id in papers:
            a = db.get_analysis(paper_id)
            if not a:
                continue
            tags = _parse(a.get('tags'), {})
            if tags:
                front_methods.update(tags.get('methods', []))
                front_problems.update(tags.get('problems', []))
                if tags.get('framework_lineage'):
                    front_lineage.add(tags['framework_lineage'])
                if tags.get('specific_domain'):
                    front_domain.add(tags['specific_domain'])

        # Count how many fronts have each tag
        for m in front_methods:
            methods_per_front[m] += 1
        for p in front_problems:
            problems_per_front[p] += 1
        for lin in front_lineage:
            lineage_per_front[lin] += 1
        for dom in front_domain:
            domain_per_front[dom] += 1

    # Tags appearing in 2+ fronts are "common" (multi-front noise)
    common = {
        'methods': [t for t, cnt in methods_per_front.most_common() if cnt >= 2],
        'problems': [t for t, cnt in problems_per_front.most_common() if cnt >= 2],
        'framework_lineage': [t for t, cnt in lineage_per_front.most_common() if cnt >= 2],
        'specific_domain': [t for t, cnt in domain_per_front.most_common() if cnt >= 2],
    }
    return common


def summarize_all_fronts(fronts: List[Dict],
                          update_db: bool = True,
                          force: bool = False) -> List[Dict]:
    """
    Enrich all fronts with name, summary, and future_directions.

    Args:
        fronts:    List of front dicts (must include front_id, snapshot_date)
        update_db: Whether to persist results to the database
        force:     If True, always regenerate via LLM even if enrichment exists

    Returns:
        Updated fronts list with name, summary, future_directions populated.
    """
    if not fronts:
        return fronts

    db = Database()
    llm_client = create_agent_client('front_summarizer')

    print(f"\n  Enriching {len(fronts)} fronts with LLM (name + summary + future directions)...")

    # Compute common tags across fronts ONCE (to avoid focusing on these in summaries)
    with db:
        common_tags = _compute_common_tags(fronts, db)

    common_summary = ", ".join(f"{k}={len(v)}" for k, v in common_tags.items() if v)
    print(f"  Common tags (appearing in 2+ fronts): {common_summary}")

    for i, front in enumerate(fronts, 1):
        fid = front['front_id']
        print(f"  [{i}/{len(fronts)}] {fid}...", end=" ", flush=True)

        # Avoid unnecessary LLM calls when enrichment is already present.
        if not force and _has_valid_enrichment(front):
            front['future_directions'] = _normalize_future_directions(front.get('future_directions'))
            print("SKIP (already enriched)")
            continue

        try:
            with db:
                enrichment = enrich_front(front, db, llm_client, common_tags)

            front.update(enrichment)
            name_preview = enrichment['name'][:60]
            print(f"OK — \"{name_preview}\"")

            if update_db:
                with db:
                    db.execute(
                        """UPDATE research_fronts
                           SET name = ?, summary = ?, future_directions = ?
                           WHERE front_id = ? AND snapshot_date = ?""",
                        (
                            enrichment['name'],
                            enrichment['summary'],
                            json.dumps(enrichment['future_directions']),
                            fid,
                            front['snapshot_date'],
                        )
                    )
                    db.commit()

        except Exception as e:
            print(f"FAILED: {e}")
            front['name'] = "[Enrichment failed]"
            front['summary'] = f"[Summary generation failed: {e}]"
            front['future_directions'] = []
            # Persist the failure marker so the DB doesn't silently keep a
            # stale or empty summary from a prior run.
            if update_db:
                try:
                    with db:
                        db.execute(
                            """UPDATE research_fronts
                               SET name = ?, summary = ?, future_directions = ?
                               WHERE front_id = ? AND snapshot_date = ?""",
                            (
                                front['name'],
                                front['summary'],
                                json.dumps(front['future_directions']),
                                fid,
                                front['snapshot_date'],
                            )
                        )
                        db.commit()
                except Exception:
                    pass  # DB update is best-effort on failure path

    return fronts
