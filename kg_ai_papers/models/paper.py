# kg_ai_papers/models/paper.py

from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple

from .section import Section
from .reference import Reference
from .concept import ConceptOccurrence


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    pdf_path: Optional[str] = None

    sections: List[Section] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)

    concepts: List[ConceptOccurrence] = field(default_factory=list)
    paper_level_concepts: List[Tuple[str, float]] = field(default_factory=list)
    embedding: Any = None  # torch.Tensor or np.ndarray
