from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

import networkx as nx
from pydantic import BaseModel, Field

from kg_ai_papers.config.settings import settings
from kg_ai_papers.graph.schema import NodeType, EdgeType
from kg_ai_papers.models.paper import Paper


# ---------------------------------------------------------------------------
# Pydantic view models used by the web API
# ---------------------------------------------------------------------------


class PaperSummary(BaseModel):
    """Minimal paper info used in influence views."""

    arxiv_id: str
    title: Optional[str] = None


class ConceptView(BaseModel):
    """View model for a single concept with a weight/score."""

    label: str
    weight: float

    @property
    def score(self) -> float:
        """Backwards-compatible alias, in case anything uses `.score`."""
        return self.weight



class InfluenceEdgeView(BaseModel):
    """View for a single influence edge (paper ↔ paper)."""

    arxiv_id: str
    title: Optional[str] = None
    influence_score: float
    similarity: float


class PaperInfluenceResult(BaseModel):
    """
    Full influence view returned by both the Python API and the FastAPI
    `/papers/{arxiv_id}` endpoint.
    """

    paper: PaperSummary
    concepts: List[ConceptView] = Field(default_factory=list)
    references: List[InfluenceEdgeView] = Field(default_factory=list)
    influenced: List[InfluenceEdgeView] = Field(default_factory=list)

    @property
    def influential_references(self) -> List[InfluenceEdgeView]:
        """Backwards-compatible alias for tests that expect this name."""
        return self.references

    @property
    def influenced_papers(self) -> List[InfluenceEdgeView]:
        """Backwards-compatible alias for tests that expect this name."""
        return self.influenced



class PaperConceptsView(BaseModel):
    """
    Optional richer representation for concepts; not used in tests but kept
    for potential future endpoints.

    NOTE: `get_paper_concepts` (Python API) returns a list[ConceptView]
    to match the unit tests.
    """

    arxiv_id: str
    title: Optional[str] = None
    paper_level_concepts: List[ConceptView] = Field(default_factory=list)
    section_concepts: Dict[str, List[ConceptView]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_paper_node_id(G: nx.MultiDiGraph, arxiv_id: str) -> str:
    """
    Resolve the *node id* for a given arxiv_id.

    We support both:
      - node id == arxiv_id
      - node has attribute arxiv_id == arxiv_id

    This is required because tests build toy graphs where node IDs may not
    literally be "p1", but they set node["arxiv_id"] = "p1".
    """
    if arxiv_id in G:
        return arxiv_id

    for node_id, data in G.nodes(data=True):
        if data.get("arxiv_id") == arxiv_id:
            return node_id

    raise ValueError(f"Paper with arxiv_id={arxiv_id} not found in graph.")


def _load_enriched_paper_by_id(arxiv_id: str, node_data: Optional[dict] = None) -> Paper:
    """
    Load an enriched paper from data/enriched/<arxiv_id>.json and
    rehydrate it into a Paper model.

    For unit tests, the enriched JSON usually doesn't exist; in that case we
    fall back to a minimal Paper built from node attributes.
    """
    path: Path = settings.enriched_dir / f"{arxiv_id}.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return Paper(**data)

    # Fallback for tests / toy graphs: construct from node_data if available
    node_data = node_data or {}
    return Paper(
        arxiv_id=arxiv_id,
        title=node_data.get("title"),
        abstract=node_data.get("abstract", ""),
        pdf_path=None,
        sections=[],
        references=[],
        concepts=[],
        paper_level_concepts=node_data.get("paper_level_concepts", []),
        embedding=None,
    )


def _compute_section_concepts(
    paper: Paper, top_per_section: int = 5
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build a mapping from section title to a few top concepts in that section.

    This is mainly for API/CLI display; we aggregate by (section_id, label).
    If the paper has no section-level concept occurrences, returns {}.
    """
    if not paper.sections or not paper.concepts:
        return {}

    # Map section_id -> section_title
    section_by_id: Dict[str, str] = {}
    for idx, s in enumerate(paper.sections):
        sid = getattr(s, "id", None) or str(idx)
        section_by_id[sid] = s.title or f"Section {idx + 1}"

    # Aggregate weights by (section_id, label)
    agg: Dict[str, Dict[str, float]] = {}
    for occ in paper.concepts:
        sid = getattr(occ, "section_id", None)
        label = getattr(occ, "label", None)
        weight = getattr(occ, "weight", None)
        if not sid or not label or weight is None:
            continue

        if sid not in agg:
            agg[sid] = {}
        agg[sid][label] = agg[sid].get(label, 0.0) + float(weight)

    # Convert to {section_title: [(label, score), ...]}
    out: Dict[str, List[Tuple[str, float]]] = {}
    for sid, label_scores in agg.items():
        section_title = section_by_id.get(sid, sid)
        items = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        out[section_title] = [(lab, float(score)) for lab, score in items[:top_per_section]]

    return out


# ---------------------------------------------------------------------------
# Public API functions used by tests and CLI
# ---------------------------------------------------------------------------


def get_paper_concepts(
    G: nx.MultiDiGraph,
    arxiv_id: str,
    top_k_concepts: int = 10,
) -> List[ConceptView]:
    """
    Return the top-N paper-level concepts for a given paper in graph G.

    Tests expect this function to return a list of objects with a `.label`
    attribute (and implicitly a score).

    Behaviour:
      * Resolve node by arxiv_id (either as node key or node["arxiv_id"])
      * Order of preference for concept source:
          1. node["paper_level_concepts"] if present
             - may be list[(label, score)] or list[ConceptView]
          2. concepts derived from PAPER_HAS_CONCEPT edges
          3. concepts loaded from enriched JSON (real pipeline)
    """
    node_id = _resolve_paper_node_id(G, arxiv_id)
    node_data = G.nodes[node_id]

    plc = node_data.get("paper_level_concepts")

    # Normalize node-stored concepts to list[(label, score)] if present
    if plc:
        if isinstance(plc[0], ConceptView):
            pairs: List[Tuple[str, float]] = [(c.label, float(c.score)) for c in plc]
        elif isinstance(plc[0], (tuple, list)) and len(plc[0]) == 2:
            pairs = [(str(pl[0]), float(pl[1])) for pl in plc]
        else:
            pairs = []
    else:
        pairs = []

    # 2) If missing or empty, derive from PAPER_HAS_CONCEPT edges in the graph
    if not pairs:
        concept_scores: Dict[str, float] = {}
        for _, concept_node, edge_data in G.out_edges(node_id, data=True):
            if edge_data.get("type") != EdgeType.PAPER_HAS_CONCEPT.value:
                continue

            cdata = G.nodes[concept_node]
            label = cdata.get("label")
            if not label:
                continue

            weight = float(edge_data.get("weight", 1.0))
            # aggregate by taking max weight per label
            concept_scores[label] = max(concept_scores.get(label, 0.0), weight)

        pairs = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)

    # 3) If still empty, fallback to enriched JSON (in case pipeline wrote it)
    if not pairs:
        paper = _load_enriched_paper_by_id(arxiv_id, node_data=node_data)
        if paper.paper_level_concepts:
            pairs = [
                (str(lbl), float(score))
                for (lbl, score) in paper.paper_level_concepts
            ]

    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k_concepts]

    # Convert to ConceptView objects to satisfy tests (expects `.label` and `.weight`)
    return [ConceptView(label=lab, weight=float(score)) for lab, score in pairs_sorted]


def get_influential_references(
    G: nx.MultiDiGraph,
    arxiv_id: str,
    top_k_references: int = 10,
) -> List[InfluenceEdgeView]:
    """
    Return the most influential references that this paper CITES.
    (Outgoing PAPER_CITES_PAPER edges.)

    Tests already pass with this shape:
      - list of objects with attributes: arxiv_id, title, influence_score, similarity
    """
    node_id = _resolve_paper_node_id(G, arxiv_id)

    refs: List[InfluenceEdgeView] = []
    for _, tgt, data in G.out_edges(node_id, data=True):
        if data.get("type") != EdgeType.PAPER_CITES_PAPER.value:
            continue

        tgt_data = G.nodes[tgt]
        refs.append(
            InfluenceEdgeView(
                arxiv_id=tgt_data.get("arxiv_id", tgt),
                title=tgt_data.get("title"),
                influence_score=float(data.get("influence_score", 0.0)),
                similarity=float(data.get("similarity", 0.0)),
            )
        )

    refs.sort(key=lambda r: r.influence_score, reverse=True)
    return refs[:top_k_references]


def get_influenced_papers(
    G: nx.MultiDiGraph,
    arxiv_id: str,
    top_k_influenced: int = 10,
) -> List[InfluenceEdgeView]:
    """
    Return papers that appear to be influenced BY this one.
    (Incoming PAPER_CITES_PAPER edges.)
    """
    node_id = _resolve_paper_node_id(G, arxiv_id)

    influenced: List[InfluenceEdgeView] = []
    for src, _, data in G.in_edges(node_id, data=True):
        if data.get("type") != EdgeType.PAPER_CITES_PAPER.value:
            continue

        src_data = G.nodes[src]
        influenced.append(
            InfluenceEdgeView(
                arxiv_id=src_data.get("arxiv_id", src),
                title=src_data.get("title"),
                influence_score=float(data.get("influence_score", 0.0)),
                similarity=float(data.get("similarity", 0.0)),
            )
        )

    influenced.sort(key=lambda r: r.influence_score, reverse=True)
    return influenced[:top_k_influenced]


def get_paper_influence_view(
    G: nx.MultiDiGraph,
    arxiv_id: str,
    top_k_concepts: int = 10,    # kept for API compatibility
    top_k_references: int = 10,
    top_k_influenced: int = 10,
) -> PaperInfluenceResult:
    """
    High-level view that bundles:
      - summary of the paper
      - its top concepts
      - the references it cites (with influence scores)
      - the papers that cite it (influenced)

    Tests expect:
      - `view.paper.arxiv_id == "p1"`
      - `len(view.concepts) == 1` in toy graph
      - `len(view.influential_references) == 1`
    """
    node_id = _resolve_paper_node_id(G, arxiv_id)
    node_data = G.nodes[node_id]

    paper_summary = PaperSummary(
        arxiv_id=node_data.get("arxiv_id", arxiv_id),
        title=node_data.get("title"),
    )

    concepts = get_paper_concepts(G, arxiv_id=arxiv_id, top_k_concepts=top_k_concepts)
    references = get_influential_references(
        G, arxiv_id=arxiv_id, top_k_references=top_k_references
    )
    influenced = get_influenced_papers(
        G, arxiv_id=arxiv_id, top_k_influenced=top_k_influenced
    )

    return PaperInfluenceResult(
        paper=paper_summary,
        concepts=concepts,
        references=references,
        influenced=influenced,
    )
