# kg_ai_papers/graph/builder.py

from __future__ import annotations

from typing import Any, Dict, List, Optional
from types import SimpleNamespace

import networkx as nx
import numpy as np
from kg_ai_papers.ingest.pipeline import IngestedPaperResult

from kg_ai_papers.nlp.concept_extraction import ConceptSummary
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

def paper_node_id(paper_id: str) -> str:
    """Return the canonical node id for a paper."""
    return f"paper:{paper_id}"


def concept_node_id(concept_key: str) -> str:
    """Return the canonical node id for a concept."""
    return f"concept:{concept_key}"

def add_concept_summaries_for_paper(
    graph: nx.MultiDiGraph,
    paper_id: str,
    concept_summaries: Dict[str, ConceptSummary],
) -> None:
    """
    Ensure a paper node exists and attach concept nodes + edges based on ConceptSummary.

    Args:
        graph: The graph to mutate.
        paper_id: Internal id for the paper (e.g. "arxiv:2401.00001").
        concept_summaries: Mapping concept_key -> ConceptSummary.
    """
    p_node = paper_node_id(paper_id)

    # Make sure the paper node exists; if your existing builder already creates it,
    # this will simply add attributes or noop.
    if p_node not in graph:
        graph.add_node(p_node, type="paper", paper_id=paper_id)

    for c_key, summary in concept_summaries.items():
        c_node = concept_node_id(c_key)

        # Create or update the concept node
        if c_node not in graph:
            graph.add_node(
                c_node,
                type="concept",
                concept_key=summary.concept_key,
                name=summary.base_name,
                kind=summary.kind,
            )
        else:
            # Optionally, update attributes on existing concept nodes
            node_data = graph.nodes[c_node]
            node_data.setdefault("type", "concept")
            node_data.setdefault("concept_key", summary.concept_key)
            node_data.setdefault("name", summary.base_name)
            if summary.kind is not None:
                node_data.setdefault("kind", summary.kind)

        # Add / annotate the edge from paper -> concept
        graph.add_edge(
            p_node,
            c_node,
            type="MENTIONS_CONCEPT",
            mentions_total=summary.mentions_total,
            mentions_by_section=summary.mentions_by_section,
            weighted_score=summary.weighted_score,
        )

def attach_concepts_to_graph(
    graph: nx.MultiDiGraph,
    paper_concept_summaries: Dict[str, Dict[str, ConceptSummary]],
) -> nx.MultiDiGraph:
    """
    Attach concept summaries for many papers.

    Args:
        graph: An existing graph (e.g., already containing Paper + citation edges).
        paper_concept_summaries: Dict mapping paper_id ->
                                 (dict mapping concept_key -> ConceptSummary).

    Returns:
        The same graph instance, for convenience.
    """
    for paper_id, summaries in paper_concept_summaries.items():
        add_concept_summaries_for_paper(graph, paper_id, summaries)
    return graph

def get_paper_concept_edges(
    graph: nx.MultiDiGraph,
    paper_id: str,
    concept_key: str,
) -> List[Dict[str, Any]]:
    """
    Return all edge attribute dicts for edges between a paper and concept node.
    """
    p_node = paper_node_id(paper_id)
    c_node = concept_node_id(concept_key)

    edge_dict = graph.get_edge_data(p_node, c_node) or {}
    return list(edge_dict.values())

def update_graph_with_ingested_paper(
    graph: nx.MultiDiGraph,
    result: Any,
) -> nx.MultiDiGraph:
    """
    Update the in-memory NetworkX graph with the output of ingestion.

    This is intentionally tolerant about the shape of `result`:

      - If `result` has a `.paper` attribute (IngestedPaperResult), we use it.
      - If `result` looks like an IngestedPaper (has `.arxiv_id` and
        `.concept_summaries` but no `.paper`), we normalize it on the fly.

    Expected fields (either on `result` or on its `.paper`):

      - paper.arxiv_id
      - concept_summaries: Dict[str, ConceptSummary]
      - references: Optional[List[str]]  (may be absent; treated as [])
    """

    # ------------------------------------------------------------------
    # Normalize `result` into a (paper-like, concept_summaries, references) triple
    # ------------------------------------------------------------------
    if hasattr(result, "paper"):
        # High-level API: result is an IngestedPaperResult
        paper: Paper = result.paper  # type: ignore[assignment]
        concept_summaries: Dict[str, ConceptSummary] = (
            getattr(result, "concept_summaries", {}) or {}
        )
        references: List[str] = list(getattr(result, "references", []) or [])
    else:
        # Lower-level API: result is an IngestedPaper from the ingestion pipeline.
        arxiv_id = getattr(result, "arxiv_id", None)
        if arxiv_id is None:
            raise ValueError(
                "update_graph_with_ingested_paper: result has no `.paper` "
                "attribute and no `.arxiv_id`; cannot normalize."
            )

        # Create a lightweight paper-like object without touching the real Paper model
        paper = SimpleNamespace(
            arxiv_id=arxiv_id,
            title=getattr(result, "title", None),
            year=getattr(result, "year", None),
        )
        concept_summaries = getattr(result, "concept_summaries", {}) or {}
        references = []  # no citation info at this layer yet

    # ------------------------------------------------------------------
    # 1) Ensure the main paper node exists
    # ------------------------------------------------------------------
    paper_node = paper.arxiv_id

    if paper_node not in graph:
        graph.add_node(
            paper_node,
            type="paper",
            arxiv_id=paper.arxiv_id,
            title=getattr(paper, "title", None),
            year=getattr(paper, "year", None),
        )
    else:
        # Update basic attributes if they are present
        graph.nodes[paper_node].update(
            {
                "title": getattr(paper, "title", graph.nodes[paper_node].get("title")),
                "year": getattr(paper, "year", graph.nodes[paper_node].get("year")),
            }
        )

    # ------------------------------------------------------------------
    # 2) Attach concept summaries as concept nodes + edges
    # ------------------------------------------------------------------
    for key, summary in concept_summaries.items():
        concept_node = f"concept::{key}"

        if concept_node not in graph:
            graph.add_node(
                concept_node,
                type="concept",
                key=summary.concept_key,
                name=summary.base_name,
                kind=summary.kind,
            )

        graph.add_edge(
            paper_node,
            concept_node,
            key=f"mentions::{paper_node}::{concept_node}",
            type="MENTIONS",
            mentions_total=summary.mentions_total,
            weighted_score=summary.weighted_score,
            mentions_by_section=summary.mentions_by_section,
        )

    # ------------------------------------------------------------------
    # 3) (Optional) Attach citation edges (paper -> referenced paper)
    # ------------------------------------------------------------------
    for ref_id in references:
        ref_node = ref_id
        if ref_node not in graph:
            graph.add_node(ref_node, type="paper", arxiv_id=ref_id)

        graph.add_edge(
            paper_node,
            ref_node,
            type="CITES",
        )

    return graph