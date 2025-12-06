# kg_ai_papers/models/reference.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Reference:
    title: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    raw: str
