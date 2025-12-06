# kg_ai_papers/api/models.py

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ConceptSummary(BaseModel):
    """
    A single concept with its importance weight for a paper.
    """
    label: str = Field(..., description="Normalized concept label.")
    weight: float = Field(..., description="Importance weight for this paper.")


class PaperSummary(BaseModel):
    """
    Basic metadata about a paper.
    """
    arxiv_id: str = Field(..., description="arXiv ID of the paper.")
    title: str = Field(..., description="Title of the paper.")
    abstract: Optional[str] = Field(
        None,
        description="Abstract of the paper, if available.",
    )


class NeighborPaperInfluence(BaseModel):
    """
    Represents how strongly a neighbor paper (referenced or citing) is related
    to the focal paper, according to your influence metric.
    """
    arxiv_id: str = Field(..., description="Neighbor paper arXiv ID.")
    title: str = Field(..., description="Neighbor paper title.")
    influence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized influence score [0,1].",
    )
    similarity: float = Field(
        ...,
        description="Raw cosine similarity between paper embeddings.",
    )


class PaperInfluenceResult(BaseModel):
    """
    High-level result: concepts + influential references + influenced papers
    for a given focal paper.
    """
    paper: PaperSummary
    concepts: List[ConceptSummary] = Field(
        default_factory=list,
        description="Top concepts for this paper.",
    )
    influential_references: List[NeighborPaperInfluence] = Field(
        default_factory=list,
        description="Papers this one builds on, ranked by influence.",
    )
    influenced_papers: List[NeighborPaperInfluence] = Field(
        default_factory=list,
        description="Papers that build on this one, ranked by influence.",
    )
