# kg_ai_papers/graph/schema.py

from enum import Enum


class NodeType(str, Enum):
    PAPER = "paper"
    CONCEPT = "concept"


class EdgeType(str, Enum):
    # Legacy / coarse paper→concept relation used by build_graph
    PAPER_HAS_CONCEPT = "PAPER_HAS_CONCEPT"

    # Paper→paper citation edges
    PAPER_CITES_PAPER = "PAPER_CITES_PAPER"

    # Fine-grained ingestion-time paper→concept mentions
    PAPER_MENTIONS_CONCEPT = "PAPER_MENTIONS_CONCEPT"

