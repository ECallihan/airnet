# kg_ai_papers/models/concept.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConceptOccurrence:
    paper_id: str
    label: str
    score: float
    source_section_title: Optional[str] = None
    source_section_level: Optional[str] = None
    position_in_section: Optional[int] = None  # rank within section
