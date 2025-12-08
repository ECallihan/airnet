# kg_ai_papers/documents.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from kg_ai_papers.tei_parser import PaperSection


@dataclass
class PaperDocument:
    """
    High-level representation of a research paper used by the KG pipeline.

    Attributes:
        paper_id: Stable internal ID (e.g., "arxiv:2401.00001").
        title: Paper title (if available).
        abstract: Abstract text (optional).
        sections: List of parsed sections from TEI (PaperSection objects).
        tei_path: Path to the TEI XML on disk (if stored).
        metadata: Extra metadata (arXiv/Scholar fields, year, authors, etc.).
    """
    paper_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    sections: List[PaperSection] = field(default_factory=list)
    tei_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """
        Convenience property: abstract + all section texts concatenated.
        """
        parts: List[str] = []
        if self.abstract:
            parts.append(self.abstract)

        for sec in self.sections:
            if sec.text:
                parts.append(sec.text)

        return "\n\n".join(parts)
