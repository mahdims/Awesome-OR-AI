"""Paper response schemas — mirror the shapes the UI's data.jsx used.

Two views:
- ``PaperListItem`` — what the Feed grid + Today list render. ~25 fields.
- ``PaperDetail`` — what the drawer reads. Adds vs_baselines, lineage etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Relevance(BaseModel):
    methodological: int = 0
    problem: int = 0
    inspirational: int = 0


class PaperListItem(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str] = []
    affiliations: str = ""
    category: str = ""
    published_date: str = ""
    priority_score: float = 0.0
    must_read: bool = False
    changes_thinking: bool = False
    team_discussion: bool = False
    relevance: Relevance = Relevance()
    brief: str = ""
    methods: List[str] = []
    problems: List[str] = []
    problem_short: str = ""
    novelty_type: str = ""
    framework_lineage: Optional[str] = None
    code_url: Optional[str] = None


class PaperDetail(PaperListItem):
    abstract: str = ""
    benchmarks: List[str] = []
    baselines: List[str] = []
    vs_baselines: Dict[str, str] = {}
    closest_prior: str = ""
    llm_model: Optional[str] = None
    new_benchmark: bool = False
    confidence_results: Optional[str] = None
    reasoning: str = ""
    # Raw JSONB fields for power-users / drawer expansion
    extras: Dict[str, Any] = {}
