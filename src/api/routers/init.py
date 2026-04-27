"""Single bootstrap endpoint.

The UI's data.jsx attached papers/subdomains/etc. to ``window.*`` synchronously
at script-load time. To keep that one-shot pattern under a real backend,
``/api/init`` returns a single payload that the new ``api_client.js`` writes
back onto window.* before the React tree mounts.

Public-ish: returns ``me=null`` and an empty queue when not logged in, so the
landing page can still render papers if we ever decide to allow anonymous read.
For M1b proper, the operator-side prereq is to log in first.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.auth.jwt import current_user_optional
from src.api.deps import get_session
from src.api.routers.me import get_follows, get_pins, get_prefs, get_queue, PaperState, PrefsResponse, IdList
from src.api.routers.papers import query_papers
from src.api.routers.subdomains import load_subdomains
from src.api.schemas.paper import PaperListItem
from src.api.schemas.subdomain import Subdomain
from src.api.schemas.user import MeResponse

router = APIRouter(prefix="/api", tags=["init"])


class InitResponse(BaseModel):
    me: Optional[MeResponse] = None
    subdomains: List[Subdomain]
    categories: List[str]
    papers: List[PaperListItem]
    queue: List[PaperState] = []
    follows: IdList = IdList(ids=[])
    pins: IdList = IdList(ids=[])
    prefs: PrefsResponse = PrefsResponse()
    # Stubs the UI consumes today; real data lands in M3/M4.
    gaps: Dict[str, Any] = {}
    signals: Dict[str, Any] = {}


@router.get("/init", response_model=InitResponse)
def init(
    user: Optional[dict] = Depends(current_user_optional),
    session: Session = Depends(get_session),
) -> InitResponse:
    sds = load_subdomains()
    categories: List[str] = []
    seen = set()
    for sd in sds:
        if sd.category and sd.category not in seen:
            seen.add(sd.category)
            categories.append(sd.category)

    # First page of recent papers — UI tabs Today / Feed render directly off this.
    papers = query_papers(session, page=1, page_size=200)

    if user is None:
        return InitResponse(
            me=None,
            subdomains=sds,
            categories=categories,
            papers=papers,
        )

    return InitResponse(
        me=MeResponse(**user),
        subdomains=sds,
        categories=categories,
        papers=papers,
        queue=get_queue(user=user, session=session),
        follows=get_follows(user=user, session=session),
        pins=get_pins(user=user, session=session),
        prefs=get_prefs(user=user, session=session),
    )
