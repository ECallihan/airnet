# kg_ai_papers/graph/builder.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    G: nx.MultiDiGraph,
    result: IngestedPaperResult,
) -> None:
    """
    Incrementally update an existing graph with the outputs from ingest_single_pdf.

    - Ensures the paper node exists (with metadata).
    - Attaches concept nodes + edges using existing ConceptSummary logic.
    - Adds citation edges to any referenced papers (creating stub paper nodes if needed).
    """
    paper = result.paper
    paper_id = paper_node_id(paper.arxiv_id)

    # 1) Ensure the main paper node exists with up-to-date attributes
    if paper_id not in G:
        G.add_node(
            paper_id,
            kind="paper",
            arxiv_id=paper.arxiv_id,
            title=getattr(paper, "title", None),
            year=getattr(paper, "year", None),
            abstract=getattr(paper, "abstract", None),
        )
    else:
        # Optionally update attributes if they were placeholders before
        G.nodes[paper_id].update(
            {
                "title": getattr(paper, "title", None),
                "year": getattr(paper, "year", None),
                "abstract": getattr(paper, "abstract", None),
            }
        )

    # 2) Attach concepts using your existing aggregation helper
    #    (this expects {arxiv_id: Dict[str, ConceptSummary]})
    from kg_ai_papers.graph.builder import attach_concepts_to_graph  # local import to avoid cycles

    attach_concepts_to_graph(
        G,
        {paper.arxiv_id: result.concept_summaries},
    )

    # 3) Add citation edges for each reference
    for ref_arxiv in result.references:
        ref_id = paper_node_id(ref_arxiv)

        if ref_id not in G:
            # Create a stub node; we may enrich it later if that paper is ingested
            G.add_node(
                ref_id,
                kind="paper",
                arxiv_id=ref_arxiv,
            )

        # Edge: paper -> reference (paper cites ref)
        G.add_edge(
            paper_id,
            ref_id,
            key=f"cites:{paper.arxiv_id}->{ref_arxiv}",
            kind="cites",
            weight=1.0,
        )

def update_graph_with_ingested_paper(
    graph: nx.MultiDiGraph,
    result: "IngestedPaperResult",
) -> nx.MultiDiGraph:
    """
    Merge a single IngestedPaperResult into an existing NetworkX graph.

    This function bridges the ingestion pipeline and the graph layer:

    - Ensures a *base* paper node exists (using the same schema as build_graph),
      keyed by `paper.arxiv_id`.
    - Attaches concept summaries via the concept-layer helpers
      (paper_node_id / concept_node_id, MENTIONS_CONCEPT edges).
    - Adds citation edges from the ingested paper to any referenced papers,
      using the PAPER_CITES_PAPER edge type.

    Parameters
    ----------
    graph:
        The graph to mutate (nx.MultiDiGraph).
    result:
        An IngestedPaperResult (or any duck-typed object with
        `.paper`, `.concept_summaries`, `.references` attributes).

    Returns
    -------
    nx.MultiDiGraph
        The same graph instance, for convenience.
    """
    # 1) Ensure the main paper node exists (unprefixed arxiv_id, same as build_graph)
    paper = result.paper
    paper_node = _get_or_create_paper_node(graph, paper)  # node id == paper.arxiv_id

    # 2) Attach concept summaries using the concept-layer helpers
    #    Here we treat the paper_id as the arxiv_id; this will create/attach
    #    a "paper:{arxiv_id}" node in the concept subgraph.
    if getattr(result, "concept_summaries", None):
        add_concept_summaries_for_paper(
            graph,
            paper_id=paper.arxiv_id,
            concept_summaries=result.concept_summaries,
        )

    # 3) Add citation edges for any references reported by ingestion
    for cited_id in getattr(result, "references", []) or []:
        if not cited_id:
            continue

        # Ensure the cited paper node exists in the *base* graph schema
        if cited_id in graph:
            cited_node = cited_id
        else:
            graph.add_node(
                cited_id,
                type=NodeType.PAPER.value,
                arxiv_id=cited_id,
                title="",
                abstract="",
            )
            cited_node = cited_id

        # For now we don't have embeddings here, so influence/similarity are 0.0
        _ensure_edge(
            graph,
            paper_node,
            cited_node,
            EdgeType.PAPER_CITES_PAPER,
            influence_score=0.0,
            similarity=0.0,
        )

    return graph
