# kg_ai_papers/models/concept.py

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Concept(BaseModel):
    """
    Represents a weighted concept/keyword extracted from text.

    Fields
    ------
    label:
        The text of the concept / keyphrase.
    weight:
        Strength or relevance score (e.g. from KeyBERT), usually in [0, 1].
    source:
        Optional tag like "paper-level" or "section-level".
    section_id:
        Optional identifier of the section this concept came from.
    """

    label: str
    weight: float
    source: Optional[str] = None
    section_id: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.label, round(self.weight, 4), self.source, self.section_id))


class ConceptOccurrence(Concept):
    """
    A specific occurrence of a concept in a document.

    Extends Concept with optional positional + paper information.
    Tests that import this type generally just need it to exist and
    behave like a Concept.
    """

    paper_id: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
