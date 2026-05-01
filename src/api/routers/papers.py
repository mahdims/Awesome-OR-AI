"""Papers router — list, search, single-paper detail.

Reads from `paper_analyses` (and optionally enriches via the `papers` intake
table for code_url fallback). JSONB fields come back as dicts via psycopg's
adapter; we project the few keys the UI's data.jsx exposed today.

Search in M1b is lexical: ILIKE over title+brief+abstract. Semantic search
(embedding nearest-neighbor) lands in M1c.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.deps import get_session
from src.api.schemas.paper import PaperDetail, PaperListItem, Relevance

router = APIRouter(prefix="/api", tags=["papers"])


# --- helpers ---------------------------------------------------------------


def _row_to_list_item(row: Dict[str, Any]) -> PaperListItem:
    rel = row.get("relevance") or {}
    sig = row.get("significance") or {}
    tags = row.get("tags") or {}
    problem = row.get("problem") or {}
    lineage = row.get("lineage") or {}
    artifacts = row.get("artifacts") or {}

    pub = row.get("published_date")
    pub_str = pub.isoformat() if hasattr(pub, "isoformat") else (pub or "")

    return PaperListItem(
        arxiv_id=row["arxiv_id"],
        title=row.get("title") or "",
        authors=row.get("authors") or [],
        affiliations=row.get("affiliations") or "",
        category=row.get("category") or "",
        published_date=pub_str,
        priority_score=float(
            (rel.get("methodological", 0) + rel.get("problem", 0) + rel.get("inspirational", 0))
        ),
        must_read=bool(sig.get("must_read")),
        changes_thinking=bool(sig.get("changes_thinking")),
        team_discussion=bool(sig.get("team_discussion")),
        relevance=Relevance(
            methodological=int(rel.get("methodological", 0)),
            problem=int(rel.get("problem", 0)),
            inspirational=int(rel.get("inspirational", 0)),
        ),
        brief=row.get("brief") or "",
        methods=tags.get("methods") or [],
        problems=tags.get("problems") or [],
        problem_short=problem.get("short") or "",
        novelty_type=lineage.get("novelty_type") or "",
        framework_lineage=tags.get("framework_lineage"),
        code_url=artifacts.get("code_url") or None,
    )


def _row_to_detail(row: Dict[str, Any]) -> PaperDetail:
    base = _row_to_list_item(row).model_dump()
    sig = row.get("significance") or {}
    exp = row.get("experiments") or {}
    res = row.get("results") or {}
    methodology = row.get("methodology") or {}
    lineage = row.get("lineage") or {}
    artifacts = row.get("artifacts") or {}

    base.update(
        {
            "abstract": row.get("abstract") or "",
            "benchmarks": exp.get("benchmarks") or [],
            "baselines": exp.get("baselines") or [],
            "vs_baselines": (res.get("vs_baselines") or {})
            if isinstance(res.get("vs_baselines"), dict)
            else {},
            "closest_prior": lineage.get("closest_prior_work") or "",
            "llm_model": methodology.get("llm_model_used") or None,
            "new_benchmark": bool(artifacts.get("new_benchmark")),
            "confidence_results": (row.get("reader_confidence") or {}).get("results"),
            "reasoning": sig.get("reasoning") or "",
            "extras": {
                "extensions": row.get("extensions") or {},
                "methods_confidence": row.get("methods_confidence") or {},
            },
        }
    )
    return PaperDetail(**base)


# --- endpoints -------------------------------------------------------------


# Columns _row_to_list_item actually consumes. Avoids `SELECT *` pulling the
# vector(768) embedding + the heavy methods_confidence / experiments / results
# / abstract blobs that the list view never serializes — meaningful overhead
# on /api/init's 200-row default.
_LIST_COLUMNS = (
    "arxiv_id, title, authors, affiliations, category, published_date, "
    "relevance, significance, tags, problem, lineage, artifacts, brief"
)


def query_papers(
    session: Session,
    q: Optional[str] = None,
    category: Optional[str] = None,
    must_read: Optional[bool] = None,
    page: int = 1,
    page_size: int = 50,
) -> List[PaperListItem]:
    """Plain-Python query builder. Call this from /api/init and other routers."""
    where = ["is_relevant = TRUE"]
    params: Dict[str, Any] = {}
    if q:
        where.append("(title ILIKE :q OR brief ILIKE :q OR abstract ILIKE :q)")
        params["q"] = f"%{q}%"
    if category:
        where.append("category = :category")
        params["category"] = category
    if must_read is True:
        where.append("(significance->>'must_read')::bool = TRUE")

    params["limit"] = page_size
    params["offset"] = (page - 1) * page_size

    sql = text(
        f"""
        SELECT {_LIST_COLUMNS}
        FROM paper_analyses
        WHERE {' AND '.join(where)}
        ORDER BY published_date DESC NULLS LAST, arxiv_id
        LIMIT :limit OFFSET :offset
        """
    )
    rows = session.execute(sql, params).mappings().all()
    return [_row_to_list_item(dict(r)) for r in rows]


@router.get("/papers", response_model=List[PaperListItem])
def list_papers(
    q: Optional[str] = Query(None, description="Lexical search term (ILIKE on title+brief+abstract)."),
    category: Optional[str] = None,
    must_read: Optional[bool] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> List[PaperListItem]:
    return query_papers(
        session, q=q, category=category, must_read=must_read, page=page, page_size=page_size,
    )


@router.get("/papers/{arxiv_id}", response_model=PaperDetail)
def get_paper(arxiv_id: str, session: Session = Depends(get_session)) -> PaperDetail:
    row = (
        session.execute(
            text("SELECT * FROM paper_analyses WHERE arxiv_id = :id"),
            {"id": arxiv_id},
        )
        .mappings()
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return _row_to_detail(dict(row))
