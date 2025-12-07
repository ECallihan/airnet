# kg_ai_papers/graph/builder.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from kg_ai_papers.graph.schema import EdgeType, NodeType
from kg_ai_papers.models.paper import Paper


def _get_or_create_paper_node(G: nx.MultiDiGraph, paper: Paper) -> str:
    """
    Ensure there's a node for this paper and return its node id.

    Uses arxiv_id as the node id in the real graph.
    """
    node_id = paper.arxiv_id
    attrs = {
        "type": NodeType.PAPER.value,
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract,
        # this makes API calls work without hitting disk
        "paper_level_concepts": paper.paper_level_concepts or [],
    }

    if node_id in G:
        G.nodes[node_id].update(attrs)
    else:
        G.add_node(node_id, **attrs)

    return node_id



def _extract_concept_label(concept: Any) -> str:
    """
    Extract a concept label from either:
      - an object with .label, or
      - a (label, weight) tuple/list.
    """
    if hasattr(concept, "label"):
        label = getattr(concept, "label") or ""
    elif isinstance(concept, (tuple, list)) and concept:
        label = concept[0]
    else:
        label = ""
    return (label or "").strip()


def _extract_concept_weight(concept: Any) -> float:
    """
    Extract a concept weight from either:
      - an object with .weight, or
      - a (label, weight) tuple/list.
    """
    if hasattr(concept, "weight"):
        w = getattr(concept, "weight", None)
    elif isinstance(concept, (tuple, list)) and len(concept) > 1:
        w = concept[1]
    else:
        w = None

    try:
        return float(w)
    except (TypeError, ValueError):
        return 1.0


def _get_or_create_concept_node(G: nx.MultiDiGraph, concept: Any) -> Optional[str]:
    """
    Ensure there's a node for this concept label and return its node id.

    Node id is derived from the label. If no valid label found, returns None.
    """
    label = _extract_concept_label(concept)
    if not label:
        return None

    node_id = f"concept:{label.lower()}"
    if node_id in G:
        # Ensure attributes are consistent
        G.nodes[node_id].update(
            {
                "type": NodeType.CONCEPT.value,
                "label": label,
            }
        )
    else:
        G.add_node(
            node_id,
            type=NodeType.CONCEPT.value,
            label=label,
        )
    return node_id


def _ensure_edge(
    G: nx.MultiDiGraph,
    src: str,
    dst: str,
    edge_type: EdgeType,
    **attrs,
) -> None:
    """
    Add an edge of a given type if it doesn't exist yet (based on src, dst, type).
    If it exists, update its attributes.
    """
    for _, v, data in G.edges(src, data=True):
        if v == dst and data.get("type") == edge_type.value:
            data.update(attrs)
            return

    G.add_edge(src, dst, type=edge_type.value, **attrs)


def _cosine_similarity(v1, v2) -> float:
    """
    Cosine similarity between two vectors (lists, numpy arrays, or tensors).
    """
    v1_arr = np.asarray(v1, dtype=float)
    v2_arr = np.asarray(v2, dtype=float)
    denom = (np.linalg.norm(v1_arr) * np.linalg.norm(v2_arr)) + 1e-8
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1_arr, v2_arr) / denom)


def build_graph(
    papers: List[Paper],
    citation_map: Optional[Dict[str, List[str]]] = None,
    existing_graph: Optional[nx.MultiDiGraph] = None,
) -> nx.MultiDiGraph:
    """
    Build or update the MultiDiGraph from a list of enriched Papers.

    Parameters
    ----------
    papers:
        List of Paper objects. Each may have:
          - paper.paper_level_concepts: iterable of either
              * objects with .label and .weight, or
              * (label, weight) tuples
          - paper.references: optional iterable of reference-like objects.

    citation_map:
        Optional dict mapping paper_id -> list of cited paper_ids.
        This is used by tests (and can supplement parsed references).

    existing_graph:
        If provided, the graph is updated in-place and returned. Otherwise,
        a new MultiDiGraph is created.

    Returns
    -------
    nx.MultiDiGraph
        The resulting graph.
    """
    G = existing_graph or nx.MultiDiGraph()
    citation_map = citation_map or {}

    # Index papers by arxiv_id for quick lookup (for influence_score)
    paper_index: Dict[str, Paper] = {p.arxiv_id: p for p in papers}

    # 1) Ensure paper nodes & paper -> concept edges
    for paper in papers:
        paper_node = _get_or_create_paper_node(G, paper)

        # Paper-level concepts
        if getattr(paper, "paper_level_concepts", None):
            for concept in paper.paper_level_concepts:
                concept_node = _get_or_create_concept_node(G, concept)
                if concept_node is None:
                    continue
                weight = _extract_concept_weight(concept)
                _ensure_edge(
                    G,
                    paper_node,
                    concept_node,
                    EdgeType.PAPER_HAS_CONCEPT,
                    weight=weight,
                )

    # 2) Citation edges based on citation_map, with influence_score
    for paper in papers:
        src_id = paper.arxiv_id
        paper_node = src_id  # same node id as before

        for cited_id in citation_map.get(src_id, []):
            # Make sure cited paper node exists
            if cited_id in G:
                cited_node = cited_id
            else:
                # Create a bare paper node with no title/abstract if necessary
                G.add_node(
                    cited_id,
                    type=NodeType.PAPER.value,
                    arxiv_id=cited_id,
                    title="",
                    abstract="",
                )
                cited_node = cited_id

            # Compute influence_score from embeddings when available
            src_paper = paper_index.get(src_id)
            dst_paper = paper_index.get(cited_id)
            if src_paper is not None and dst_paper is not None:
                emb_src = getattr(src_paper, "embedding", None)
                emb_dst = getattr(dst_paper, "embedding", None)
            else:
                emb_src = None
                emb_dst = None

            if emb_src is not None and emb_dst is not None:
                sim = _cosine_similarity(emb_src, emb_dst)
            else:
                sim = 0.0

            _ensure_edge(
                G,
                paper_node,
                cited_node,
                EdgeType.PAPER_CITES_PAPER,
                influence_score=float(sim),
                similarity=float(sim),
            )

    # (Optionally, we could also incorporate paper.references into citations
    #  for enriched data; citation_map is enough for tests.)

    return G
