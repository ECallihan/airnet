from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

from kg_ai_papers.graph.builder import paper_node_id


# ---------- Small view models ----------

@dataclass
class PaperSummary:
    # Tests expect view.paper.arxiv_id
    # FastAPI response model expects .title as well
    arxiv_id: str
    title: str


@dataclass
class PaperConceptView:
    # Tests & web schema expect these names
    label: str
    weight: float

    # Extra fields (useful internally, but optional from tests' POV)
    mentions_total: int = 0
    mentions_by_section: Dict[str, int] = field(default_factory=dict)


@dataclass
class ReferenceInfluenceView:
    # Tests expect .arxiv_id
    arxiv_id: str
    influence_score: float

    # Extra influence features (optional)
    num_shared_concepts: int = 0
    jaccard_weighted: float = 0.0
    jaccard_unweighted: float = 0.0


@dataclass
class PaperInfluenceView:
    paper: PaperSummary
    concepts: List[PaperConceptView]
    influential_references: List[ReferenceInfluenceView]
    influenced_papers: List[ReferenceInfluenceView]


# ---------- Helpers ----------

def _find_paper_node(graph: nx.MultiDiGraph, arxiv_id: str) -> Optional[Any]:
    """
    Robustly find the node corresponding to this paper id in different graph flavors:
      - Node id == paper_node_id(arxiv_id) (e.g. "paper:p1")
      - Node id == arxiv_id
      - Node with node['arxiv_id'] == arxiv_id
    """
    # 1) Preferred: use builder's convention
    node = paper_node_id(arxiv_id)
    if node in graph:
        return node

    # 2) Raw id
    if arxiv_id in graph:
        return arxiv_id

    # 3) Attr-based lookup
    for n, attrs in graph.nodes(data=True):
        if attrs.get("arxiv_id") == arxiv_id or attrs.get("paper_id") == arxiv_id:
            return n

    return None


def _make_concept_view(
    graph: nx.MultiDiGraph,
    concept_node: Any,
    edge_data: Dict[str, Any],
) -> PaperConceptView:
    c_attrs = graph.nodes[concept_node]

    # Prefer an explicit 'label' if present (used in toy test graph),
    # fall back to 'name', then concept_node itself.
    label = (
        c_attrs.get("label")
        or c_attrs.get("name")
        or c_attrs.get("concept_key")
        or concept_node
    )

    # Prefer our more detailed 'weighted_score' if present,
    # otherwise fall back to a simple 'weight' or default 1.0.
    weight = edge_data.get("weighted_score")
    if weight is None:
        weight = edge_data.get("weight", 1.0)

    mentions_total = int(edge_data.get("mentions_total", 1))
    mentions_by_section = edge_data.get("mentions_by_section", {}) or {}

    return PaperConceptView(
        label=str(label),
        weight=float(weight),
        mentions_total=mentions_total,
        mentions_by_section=mentions_by_section,
    )



def _make_reference_view(
    graph: nx.MultiDiGraph,
    other_node: Any,
    edge_data: Dict[str, Any],
) -> ReferenceInfluenceView:
    attrs = graph.nodes[other_node]
    arxiv_id = attrs.get("arxiv_id") or attrs.get("paper_id") or other_node

    return ReferenceInfluenceView(
        arxiv_id=str(arxiv_id),
        influence_score=float(edge_data.get("influence_score", 0.0)),
        num_shared_concepts=int(edge_data.get("influence_num_shared_concepts", 0)),
        jaccard_weighted=float(edge_data.get("influence_jaccard_weighted", 0.0)),
        jaccard_unweighted=float(edge_data.get("influence_jaccard_unweighted", 0.0)),
    )


# ---------- Public query functions used by tests & web app ----------

def get_paper_concepts(
    graph: nx.MultiDiGraph,
    arxiv_id: str,
    top_k: Optional[int] = None,
) -> List[PaperConceptView]:
    """
    Return concept views for a paper.

    Primary mode:
        Uses MENTIONS_CONCEPT edges created by the concept pipeline.

    Fallback (for toy graphs used in tests):
        Any neighbor whose node type looks like a concept is treated as a concept.
    """
    p_node = _find_paper_node(graph, arxiv_id)
    if p_node is None:
        return []

    concepts: List[PaperConceptView] = []

    # ---- Primary: MENTIONS_CONCEPT edges ----
    for _, c_node, key, data in graph.out_edges(p_node, keys=True, data=True):
        etype = str(data.get("type", "")).upper()
        if etype == "MENTIONS_CONCEPT":
            concepts.append(_make_concept_view(graph, c_node, data))

    # ---- Fallback: generic concept neighbors (for tests' toy graph) ----
    if not concepts:
        for _, c_node, key, data in graph.out_edges(p_node, keys=True, data=True):
            c_attrs = graph.nodes[c_node]
            node_type = str(c_attrs.get("type", "")).lower()
            kind = str(c_attrs.get("kind", "")).lower()

            if node_type == "concept" or "concept" in kind:
                concepts.append(_make_concept_view(graph, c_node, data))

    concepts.sort(
        key=lambda c: getattr(c, "weighted_score", getattr(c, "weight", 0.0)),
        reverse=True,
    )
    if top_k is not None:
        concepts = concepts[:top_k]
    return concepts


def get_influential_references(
    graph: nx.MultiDiGraph,
    arxiv_id: str,
    top_k: int = 10,
) -> List[ReferenceInfluenceView]:
    """
    Papers this one cites (outgoing citation edges).

    Primary mode:
        Edges with type="CITES".

    Fallback:
        Any outgoing edge from this paper to another paper-like node.
    """
    p_node = _find_paper_node(graph, arxiv_id)
    if p_node is None:
        return []

    refs: List[ReferenceInfluenceView] = []

    # ---- Primary: CITES edges ----
    for _, v, key, data in graph.out_edges(p_node, keys=True, data=True):
        etype = str(data.get("type", "")).upper()
        if etype == "CITES":
            refs.append(_make_reference_view(graph, v, data))

    # ---- Fallback: any paper-like neighbor ----
    if not refs:
        for _, v, key, data in graph.out_edges(p_node, keys=True, data=True):
            v_attrs = graph.nodes[v]
            v_type = str(v_attrs.get("type", "")).lower()
            if v_type == "paper" or "arxiv_id" in v_attrs or "paper_id" in v_attrs:
                refs.append(_make_reference_view(graph, v, data))

    refs.sort(
        key=lambda r: (r.influence_score, r.num_shared_concepts),
        reverse=True,
    )
    return refs[:top_k]


def get_influenced_papers(
    graph: nx.MultiDiGraph,
    arxiv_id: str,
    top_k: int = 10,
) -> List[ReferenceInfluenceView]:
    """
    Papers that cite this one (incoming citation edges).
    """
    p_node = _find_paper_node(graph, arxiv_id)
    if p_node is None:
        return []

    influenced: List[ReferenceInfluenceView] = []

    # ---- Primary: CITES edges ----
    for u, _, key, data in graph.in_edges(p_node, keys=True, data=True):
        etype = str(data.get("type", "")).upper()
        if etype == "CITES":
            influenced.append(_make_reference_view(graph, u, data))

    # ---- Fallback: any paper-like neighbor via incoming edges ----
    if not influenced:
        for u, _, key, data in graph.in_edges(p_node, keys=True, data=True):
            u_attrs = graph.nodes[u]
            u_type = str(u_attrs.get("type", "")).lower()
            if u_type == "paper" or "arxiv_id" in u_attrs or "paper_id" in u_attrs:
                influenced.append(_make_reference_view(graph, u, data))

    influenced.sort(
        key=lambda r: (r.influence_score, r.num_shared_concepts),
        reverse=True,
    )
    return influenced[:top_k]


def get_paper_influence_view(
    graph: nx.MultiDiGraph,
    arxiv_id: str,
    *,
    top_k_concepts: int = 10,
    top_k_references: int = 10,
    top_k_influenced: int = 10,
) -> PaperInfluenceView:
    """
    High-level influence view:
      - `paper`: summary (arxiv_id + title)
      - `concepts`: top concepts
      - `influential_references`: what it builds on
      - `influenced_papers`: who builds on it
    """
    p_node = _find_paper_node(graph, arxiv_id)
    if p_node is not None:
        node_attrs = graph.nodes[p_node]
        title = node_attrs.get("title", arxiv_id)
    else:
        title = arxiv_id

    paper = PaperSummary(arxiv_id=arxiv_id, title=title)

    concepts = get_paper_concepts(graph, arxiv_id, top_k=top_k_concepts)
    influential_refs = get_influential_references(
        graph, arxiv_id, top_k=top_k_references
    )
    influenced_papers = get_influenced_papers(
        graph, arxiv_id, top_k=top_k_influenced
    )

    return PaperInfluenceView(
        paper=paper,
        concepts=concepts,
        influential_references=influential_refs,
        influenced_papers=influenced_papers,
    )
