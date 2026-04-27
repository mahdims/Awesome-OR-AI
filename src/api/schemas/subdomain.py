"""Subdomain schemas. Loaded from research_config/subdomains.yaml at startup."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Subdomain(BaseModel):
    id: str
    name: str
    tagline: str = ""
    category: str = ""
    paper_count: int = 0  # M3 will start populating; M1b always 0


class SubdomainListResponse(BaseModel):
    subdomains: List[Subdomain]


class CategoryListResponse(BaseModel):
    categories: List[str]
