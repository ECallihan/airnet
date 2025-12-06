# kg_ai_papers/api/query.py

from __future__ import annotations

from typing import List, Optional

import networkx as nx

from kg_ai_papers.api.models import (
    ConceptSummary,
    PaperSummary,
    NeighborPaperInfluence,
    PaperInfluenceResult,
)
from kg_ai_papers.graph.schema import NodeType, EdgeType
from kg_ai_papers.graph.storage import load_graph


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _paper_node_id(arxiv_id: str) -> str:
    return f"paper:{arxiv_id}"


def _ensure_paper_node(G: nx.MultiDiGraph, arxiv_id: str) -> str:
    """
    Ensure that the graph has a node for this paper, and return its node id.
    Raises KeyError if not found.
    """
    node_id = _paper_node_id(arxiv_id)
    if node_id not in G:
        raise KeyError(f"Paper with arxiv_id={arxiv_id!r} not found in graph.")
    return node_id


def _get_paper_summary_from_node(G: nx.MultiDiGraph, node_id: str) -> PaperSummary:
    """
    Build a PaperSummary from a paper node.
    """
    data = G.nodes[node_id]
    if data.get("type") != NodeType.PAPER.value:
        raise ValueError(f"Node {node_id!r} is not a paper node.")

    return PaperSummary(
        arxiv_id=data.get("arxiv_id"),
        title=data.get("title", ""),
        abstract=data.get("abstract"),
    )


# ----------------------------------------------------------------------
# Public query functions
# ----------------------------------------------------------------------

def get_paper_concepts(
    G: nx.MultiDiGraph,
    arxiv_id: str,
    top_k: Optional[int] = None,
) -> List[ConceptSummary]:
    """
    Return the (optionally top_k) concepts for the given paper,
    sorted by descending weight.

    Concepts are read from PAPER_HAS_CONCEPT edges.
    """
    node_id = _ensure_paper_node(G, arxiv_id)

    concept_weights = {}
    for _, concept_node, edge_data in G.out_edges(node_id, data=True):
        if edge_data.get("type") != EdgeType.PAPER_HAS_CONCEPT.value:
            continue
        label = G.nodes[concept_node].get("label")
        if not label:
            continue
        weight = float(edge_data.get("weight", 1.0))
        concept_weights[label] = concept_weights.get(label, 0.0) + weight

    sorted_items = sorted(
        concept_weights.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    if top_k is not None:
        sorted_items = sorted_items[:top_k]

    return [
        ConceptSummary(label=label, weight=weight)
        for label, weight in sorted_items
    ]


def get_influential_references(
    G: nx.MultiDiGraph,
    arxiv_id: str,
    top_k: Optional[int] = None,
) -> List[NeighborPaperInfluence]:
    """
    Return papers that the given paper cites (outgoing edges),
    ranked by influence_score.
    """
    node_id = _ensure_paper_node(G, arxiv_id)

    neighbors: List[NeighborPaperInfluence] = []

    for _, dst, edge_data in G.out_edges(node_id, data=True):
        if edge_data.get("type") != EdgeType.PAPER_CITES_PAPER.value:
            continue

        # destination must be a paper node
        dst_data = G.nodes[dst]
        if dst_data.get("type") != NodeType.PAPER.value:
            continue

        influence = float(edge_data.get("influence_score", 0.0))
        similarity = float(edge_data.get("similarity", 0.0))

        neighbors.append(
            NeighborPaperInfluence(
                arxiv_id=dst_data.get("arxiv_id"),
                title=dst_data.get("title", ""),
                influence_score=influence,
                similarity=similarity,
            )
        )

    neighbors.sort(key=lambda x: x.influence_score, reverse=True)

    if top_k is not None:
        neighbors = neighbors[:top_k]

    return neighbors


def get_influenced_papers(
    G: nx.MultiDiGraph,
    arxiv_id: str,
    top_k: Optional[int] = None,
) -> List[NeighborPaperInfluence]:
    """
    Return papers that cite the given paper (incoming edges),
    ranked by influence_score.
    """
    node_id = _ensure_paper_node(G, arxiv_id)

    neighbors: List[NeighborPaperInfluence] = []

    for src, _, edge_data in G.in_edges(node_id, data=True):
        if edge_data.get("type") != EdgeType.PAPER_CITES_PAPER.value:
            continue

        src_data = G.nodes[src]
        if src_data.get("type") != NodeType.PAPER.value:
            continue

        influence = float(edge_data.get("influence_score", 0.0))
        similarity = float(edge_data.get("similarity", 0.0))

        neighbors.append(
            NeighborPaperInfluence(
                arxiv_id=src_data.get("arxiv_id"),
                title=src_data.get("title", ""),
                influence_score=influence,
                similarity=similarity,
            )
        )

    neighbors.sort(key=lambda x: x.influence_score, reverse=True)

    if top_k is not None:
        neighbors = neighbors[:top_k]

    return neighbors


def get_paper_influence_view(
    G: nx.MultiDiGraph,
    arxiv_id: str,
    top_k_concepts: int = 10,
    top_k_references: int = 5,
    top_k_influenced: int = 5,
) -> PaperInfluenceResult:
    """
    High-level view for a given paper:
      - top_k_concepts
      - top_k_references (papers it builds on)
      - top_k_influenced (papers that build on it)
    """
    node_id = _ensure_paper_node(G, arxiv_id)

    paper_summary = _get_paper_summary_from_node(G, node_id)
    concepts = get_paper_concepts(G, arxiv_id, top_k=top_k_concepts)
    influential_refs = get_influential_references(G, arxiv_id, top_k=top_k_references)
    influenced_papers = get_influenced_papers(G, arxiv_id, top_k=top_k_influenced)

    return PaperInfluenceResult(
        paper=paper_summary,
        concepts=concepts,
        influential_references=influential_refs,
        influenced_papers=influenced_papers,
    )


# ----------------------------------------------------------------------
# Convenience: one-shot function that loads the graph from disk
# ----------------------------------------------------------------------

def explain_paper(
    arxiv_id: str,
    graph_name: str = "graph",
    top_k_concepts: int = 10,
    top_k_references: int = 5,
    top_k_influenced: int = 5,
) -> PaperInfluenceResult:
    """
    Convenience entrypoint:

      - loads the graph from disk (via graph.storage.load_graph)
      - returns a high-level PaperInfluenceResult for the given arxiv_id

    Great for using from a CLI or notebook.
    """
    G = load_graph(name=graph_name)
    return get_paper_influence_view(
        G,
        arxiv_id=arxiv_id,
        top_k_concepts=top_k_concepts,
        top_k_references=top_k_references,
        top_k_influenced=top_k_influenced,
    )
