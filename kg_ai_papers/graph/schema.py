# kg_ai_papers/graph/schema.py

from enum import Enum


class NodeType(str, Enum):
    PAPER = "paper"
    CONCEPT = "concept"


class EdgeType(str, Enum):
    PAPER_HAS_CONCEPT = "PAPER_HAS_CONCEPT"
    PAPER_CITES_PAPER = "PAPER_CITES_PAPER"
