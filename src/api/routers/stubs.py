"""Empty / 503 stubs for features that ship in later milestones.

Keeps the UI from blowing up if it calls these, while making the deferral
explicit. Each stub returns a Retry-After header pointing to the milestone.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/api", tags=["stubs"])


@router.get("/gaps")
def list_gaps() -> dict:
    """Always [] until M3 introduces the gap detector backed by real subdomain assignment."""
    return {"gaps": []}


@router.get("/signals")
def list_signals() -> dict:
    """Always {} until M3 wires the nightly signal aggregator."""
    return {"signals": {}}


@router.post("/novelty")
def novelty_stub():
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Novelty check ships in M4 (needs embeddings + verdict eval).",
        headers={"Retry-After": "60"},
    )


@router.post("/notebooks")
def notebooks_stub():
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Notebook + audio overview ship in M2 (NotebookLM integration).",
        headers={"Retry-After": "60"},
    )
