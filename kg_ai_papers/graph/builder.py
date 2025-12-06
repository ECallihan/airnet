# kg_ai_papers/graph/builder.py

from __future__ import annotations

from typing import Dict, List, Iterable, Tuple

import networkx as nx
from sentence_transformers import util

from kg_ai_papers.config.settings import settings
from kg_ai_papers.models.paper import Paper
from kg_ai_papers.models.concept import ConceptOccurrence
from kg_ai_papers.graph.schema import NodeType, EdgeType


def _paper_node_id(arxiv_id: str) -> str:
    return f"paper:{arxiv_id}"


def _concept_node_id(label: str) -> str:
    """
    Create a simple slug-based concept node ID.
    For now just lowercased with spaces replaced; can be improved later.
    """
    slug = label.lower().strip().replace(" ", "_")
    return f"concept:{slug}"


def _add_paper_nodes(G: nx.MultiDiGraph, papers: Iterable[Paper]) -> None:
    for paper in papers:
        node_id = _paper_node_id(paper.arxiv_id)
        G.add_node(
            node_id,
            type=NodeType.PAPER.value,
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
        )


def _add_concept_nodes_and_edges(
    G: nx.MultiDiGraph,
    papers: Iterable[Paper],
) -> None:
    """
    Add concept nodes and PAPER_HAS_CONCEPT edges.
    Uses paper.paper_level_concepts if present; otherwise aggregates from occurrences.
    """
    # Track which concept labels we've already created as nodes
    concept_labels_seen = set()

    for paper in papers:
        paper_node = _paper_node_id(paper.arxiv_id)

        # Use aggregated paper-level concepts if available
        if paper.paper_level_concepts:
            concepts = paper.paper_level_concepts
        else:
            # Fallback: summarize occurrences on the fly
            concepts = _aggregate_concepts_from_occurrences(paper.concepts)

        for label, score in concepts:
            if not label:
                continue
            c_node = _concept_node_id(label)
            if c_node not in G:
                # create concept node
                G.add_node(
                    c_node,
                    type=NodeType.CONCEPT.value,
                    label=label,
                )
                concept_labels_seen.add(label)

            G.add_edge(
                paper_node,
                c_node,
                type=EdgeType.PAPER_HAS_CONCEPT.value,
                weight=float(score),
            )


def _aggregate_concepts_from_occurrences(
    occurrences: Iterable[ConceptOccurrence],
) -> List[Tuple[str, float]]:
    """
    If you don't have paper.paper_level_concepts precomputed,
    aggregate occurrences into label -> total_score and sort.
    """
    agg: Dict[str, float] = {}
    for occ in occurrences:
        if not occ.label:
            continue
        agg[occ.label] = agg.get(occ.label, 0.0) + occ.score

    return sorted(agg.items(), key=lambda x: x[1], reverse=True)


def _add_citation_edges(
    G: nx.MultiDiGraph,
    papers_by_id: Dict[str, Paper],
    citation_map: Dict[str, List[str]],
) -> None:
    """
    Add PAPER_CITES_PAPER edges with an influence_score based on cosine similarity
    between paper embeddings.

    :param papers_by_id: mapping from arxiv_id -> Paper
    :param citation_map: citing_arxiv_id -> list of cited_arxiv_ids
    """
    for citing_id, cited_ids in citation_map.items():
        citing_paper = papers_by_id.get(citing_id)
        if citing_paper is None or citing_paper.embedding is None:
            continue

        for cited_id in cited_ids:
            cited_paper = papers_by_id.get(cited_id)
            if cited_paper is None or cited_paper.embedding is None:
                continue

            sim = util.cos_sim(
                citing_paper.embedding, cited_paper.embedding
            ).item()

            # Basic normalization from [-1,1] to [0,1]
            influence_score = max(0.0, min(1.0, (sim + 1.0) / 2.0))

            if sim < settings.CITATION_SIM_MIN:
                continue

            G.add_edge(
                _paper_node_id(citing_id),
                _paper_node_id(cited_id),
                type=EdgeType.PAPER_CITES_PAPER.value,
                similarity=float(sim),
                influence_score=float(influence_score),
            )


def build_graph(
    papers: List[Paper],
    citation_map: Dict[str, List[str]],
) -> nx.MultiDiGraph:
    """
    High-level entry point:
      - Add paper nodes
      - Add concept nodes & PAPER_HAS_CONCEPT edges
      - Add PAPER_CITES_PAPER edges with influence scores
    """
    G = nx.MultiDiGraph()

    papers_by_id = {p.arxiv_id: p for p in papers}

    _add_paper_nodes(G, papers)
    _add_concept_nodes_and_edges(G, papers)
    _add_citation_edges(G, papers_by_id, citation_map)

    return G
