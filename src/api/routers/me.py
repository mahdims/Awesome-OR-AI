"""Per-user state endpoints. M1b ships /api/me only — queue/follows/pins/prefs land in week 3."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.auth.jwt import current_user
from src.api.schemas.user import MeResponse

router = APIRouter(prefix="/api", tags=["me"])


@router.get("/me", response_model=MeResponse)
def me(user: dict = Depends(current_user)) -> MeResponse:
    return MeResponse(**user)
