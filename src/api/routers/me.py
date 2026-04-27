"""Per-user state endpoints.

GET   /api/me                     -> profile
PATCH /api/me/prefs               -> update density/theme/notification_prefs
GET   /api/me/queue               -> reading state across all papers
PATCH /api/me/papers/{arxiv_id}   -> set status / notes for a paper
GET   /api/me/follows             -> followed subdomain ids
POST  /api/me/follows/{sd_id}     -> follow a subdomain (idempotent)
DELETE /api/me/follows/{sd_id}    -> unfollow
GET   /api/me/pins                -> pinned subdomain ids in order
POST  /api/me/pins/{sd_id}        -> pin (idempotent, appends to end)
DELETE /api/me/pins/{sd_id}       -> unpin
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.auth.jwt import current_user
from src.api.deps import get_session
from src.api.schemas.user import MeResponse

router = APIRouter(prefix="/api", tags=["me"])


# ---- profile ----


@router.get("/me", response_model=MeResponse)
def me(user: dict = Depends(current_user)) -> MeResponse:
    return MeResponse(**user)


# ---- prefs ----


class PrefsResponse(BaseModel):
    density: str = "balanced"
    theme: str = "default"
    notification_prefs: Dict[str, Any] = {}


class PrefsPatch(BaseModel):
    density: Optional[str] = None
    theme: Optional[str] = None
    notification_prefs: Optional[Dict[str, Any]] = None


@router.get("/me/prefs", response_model=PrefsResponse)
def get_prefs(
    user: dict = Depends(current_user),
    session: Session = Depends(get_session),
) -> PrefsResponse:
    row = (
        session.execute(
            text("SELECT density, theme, notification_prefs FROM user_prefs WHERE user_id = :uid"),
            {"uid": user["id"]},
        )
        .mappings()
        .first()
    )
    return PrefsResponse(**dict(row)) if row else PrefsResponse()


@router.patch("/me/prefs", response_model=PrefsResponse)
def update_prefs(
    payload: PrefsPatch,
    user: dict = Depends(current_user),
    session: Session = Depends(get_session),
) -> PrefsResponse:
    from psycopg.types.json import Jsonb

    fields = payload.model_dump(exclude_unset=True)
    notif = fields.get("notification_prefs")
    if notif is not None:
        fields["notification_prefs"] = Jsonb(notif)

    # Upsert with whichever keys were provided.
    base = {"user_id": user["id"], **fields}
    cols = list(base.keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "user_id") or "user_id = user_prefs.user_id"
    sql = text(
        f"INSERT INTO user_prefs ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (user_id) DO UPDATE SET {update_clause}, updated_at = now()"
    )
    session.execute(sql, base)

    return get_prefs(user=user, session=session)


# ---- paper state / queue ----


PaperStatus = Literal["unread", "reading", "read", "discarded"]


class PaperState(BaseModel):
    paper_id: str
    status: PaperStatus
    notes: Optional[str] = None
    updated_at: Optional[str] = None


class PaperStatePatch(BaseModel):
    status: Optional[PaperStatus] = None
    notes: Optional[str] = None


@router.get("/me/queue", response_model=List[PaperState])
def get_queue(
    user: dict = Depends(current_user),
    session: Session = Depends(get_session),
) -> List[PaperState]:
    rows = (
        session.execute(
            text(
                """SELECT paper_id, status, notes, updated_at
                   FROM user_paper_state
                   WHERE user_id = :uid
                   ORDER BY updated_at DESC"""
            ),
            {"uid": user["id"]},
        )
        .mappings()
        .all()
    )
    out = []
    for r in rows:
        d = dict(r)
        ts = d.get("updated_at")
        d["updated_at"] = ts.isoformat() if ts is not None else None
        out.append(PaperState(**d))
    return out


@router.patch("/me/papers/{arxiv_id}", response_model=PaperState)
def update_paper_state(
    arxiv_id: str,
    payload: PaperStatePatch,
    user: dict = Depends(current_user),
    session: Session = Depends(get_session),
) -> PaperState:
    fields = payload.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="Provide at least one of status/notes")

    # Status default for first-time insert.
    insert_status = fields.get("status", "unread")
    insert_notes = fields.get("notes")

    sql = text(
        """
        INSERT INTO user_paper_state (user_id, paper_id, status, notes)
        VALUES (:uid, :pid, :status, :notes)
        ON CONFLICT (user_id, paper_id) DO UPDATE SET
            status     = COALESCE(:status_or_null, user_paper_state.status),
            notes      = COALESCE(:notes_or_null, user_paper_state.notes),
            updated_at = now()
        RETURNING paper_id, status, notes, updated_at
        """
    )
    row = (
        session.execute(
            sql,
            {
                "uid": user["id"],
                "pid": arxiv_id,
                "status": insert_status,
                "notes": insert_notes,
                "status_or_null": fields.get("status"),
                "notes_or_null": fields.get("notes"),
            },
        )
        .mappings()
        .first()
    )
    assert row is not None
    d = dict(row)
    ts = d.get("updated_at")
    d["updated_at"] = ts.isoformat() if ts is not None else None
    return PaperState(**d)


# ---- follows + pins ----


class IdList(BaseModel):
    ids: List[str]


@router.get("/me/follows", response_model=IdList)
def get_follows(
    user: dict = Depends(current_user),
    session: Session = Depends(get_session),
) -> IdList:
    rows = session.execute(
        text("SELECT subdomain_id FROM user_follows WHERE user_id = :uid ORDER BY created_at"),
        {"uid": user["id"]},
    ).all()
    return IdList(ids=[r[0] for r in rows])


@router.post("/me/follows/{subdomain_id}", status_code=status.HTTP_204_NO_CONTENT)
def follow(
    subdomain_id: str,
    user: dict = Depends(current_user),
    session: Session = Depends(get_session),
):
    session.execute(
        text(
            "INSERT INTO user_follows (user_id, subdomain_id) VALUES (:uid, :sd) "
            "ON CONFLICT DO NOTHING"
        ),
        {"uid": user["id"], "sd": subdomain_id},
    )
    return None


@router.delete("/me/follows/{subdomain_id}", status_code=status.HTTP_204_NO_CONTENT)
def unfollow(
    subdomain_id: str,
    user: dict = Depends(current_user),
    session: Session = Depends(get_session),
):
    session.execute(
        text("DELETE FROM user_follows WHERE user_id = :uid AND subdomain_id = :sd"),
        {"uid": user["id"], "sd": subdomain_id},
    )
    return None


@router.get("/me/pins", response_model=IdList)
def get_pins(
    user: dict = Depends(current_user),
    session: Session = Depends(get_session),
) -> IdList:
    rows = session.execute(
        text(
            "SELECT subdomain_id FROM user_pins WHERE user_id = :uid "
            "ORDER BY position, created_at"
        ),
        {"uid": user["id"]},
    ).all()
    return IdList(ids=[r[0] for r in rows])


@router.post("/me/pins/{subdomain_id}", status_code=status.HTTP_204_NO_CONTENT)
def pin(
    subdomain_id: str,
    user: dict = Depends(current_user),
    session: Session = Depends(get_session),
):
    next_pos = session.execute(
        text("SELECT COALESCE(MAX(position), 0) + 1 FROM user_pins WHERE user_id = :uid"),
        {"uid": user["id"]},
    ).scalar_one()
    session.execute(
        text(
            "INSERT INTO user_pins (user_id, subdomain_id, position) "
            "VALUES (:uid, :sd, :pos) ON CONFLICT DO NOTHING"
        ),
        {"uid": user["id"], "sd": subdomain_id, "pos": next_pos},
    )
    return None


@router.delete("/me/pins/{subdomain_id}", status_code=status.HTTP_204_NO_CONTENT)
def unpin(
    subdomain_id: str,
    user: dict = Depends(current_user),
    session: Session = Depends(get_session),
):
    session.execute(
        text("DELETE FROM user_pins WHERE user_id = :uid AND subdomain_id = :sd"),
        {"uid": user["id"], "sd": subdomain_id},
    )
    return None
