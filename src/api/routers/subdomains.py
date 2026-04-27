"""Subdomains router.

Loads research_config/subdomains.yaml at module import; no DB query needed
in M1b since per-paper assignment lands in M3.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from fastapi import APIRouter, HTTPException

from src.api.schemas.subdomain import (
    CategoryListResponse,
    Subdomain,
    SubdomainListResponse,
)

router = APIRouter(prefix="/api", tags=["subdomains"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUBDOMAINS_YAML = PROJECT_ROOT / "research_config" / "subdomains.yaml"


@lru_cache(maxsize=1)
def load_subdomains() -> List[Subdomain]:
    if not SUBDOMAINS_YAML.exists():
        return []
    with open(SUBDOMAINS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [Subdomain(**entry) for entry in data.get("subdomains", [])]


@router.get("/subdomains", response_model=SubdomainListResponse)
def list_subdomains() -> SubdomainListResponse:
    return SubdomainListResponse(subdomains=load_subdomains())


@router.get("/subdomains/{subdomain_id}", response_model=Subdomain)
def get_subdomain(subdomain_id: str) -> Subdomain:
    for sd in load_subdomains():
        if sd.id == subdomain_id:
            return sd
    raise HTTPException(status_code=404, detail="Subdomain not found")


@router.get("/categories", response_model=CategoryListResponse)
def list_categories() -> CategoryListResponse:
    """Return the 3 categories implied by the subdomain catalog (preserves the
    YAML's order of first-appearance)."""
    seen, ordered = set(), []
    for sd in load_subdomains():
        if sd.category and sd.category not in seen:
            seen.add(sd.category)
            ordered.append(sd.category)
    return CategoryListResponse(categories=ordered)
