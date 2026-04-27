"""User-facing schemas for /api/me etc."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class MeResponse(BaseModel):
    id: int
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: str = "member"
