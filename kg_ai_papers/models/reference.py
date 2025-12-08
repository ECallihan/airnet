# kg_ai_papers/models/reference.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Reference:
    """
    Simple representation of a single bibliography entry (a cited work).

    This is intentionally general:
    - It works for arXiv and non-arXiv references.
    - It captures key IDs (arxiv_id, DOI) and some useful metadata.
    - `key` is a stable identifier you can use as a node id if needed.
    """

    # A stable identifier for this reference within the graph.
    # Prefer arxiv_id, then DOI, then a title/year fallback.
    key: str

    # Full raw citation text as extracted from TEI
    raw: str

    # Optional structured fields
    title: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    authors: List[str] = field(default_factory=list)

    # Placeholder for future influence modeling
    influence_weight: float = 1.0
