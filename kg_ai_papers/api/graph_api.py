from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from kg_ai_papers.graph.io import load_graph

# -----------------------------------------------------------------------------
# Settings / config
# -----------------------------------------------------------------------------

DEFAULT_GRAPH_PATH = Path("data/graphs/airnet.gpickle")


def _get_graph_path() -> Path:
    """
    Locate the on-disk graph file. For now we just use a fixed default path.
    In the future this could read from env vars or config.
    """
    return DEFAULT_GRAPH_PATH


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title="AI Papers Graph API", version="0.1.0")


# -----------------------------------------------------------------------------
# Graph loading and helpers
# -----------------------------------------------------------------------------

def get_graph() -> nx.MultiDiGraph:
    """
    Load the NetworkX graph from disk.

    In tests, this function is monkeypatched to return an in-memory graph,
    so the implementation here is only used in "real" runs.
    """
    path = _get_graph_path()
    return load_graph(path)


def _find_paper_node(G: nx.MultiDiGraph, arxiv_id: str) -> Optional[str]:
    """
    Locate the node id of a paper with the given arxiv_id attribute.
    """
    for node_id, data in G.nodes(data=True):
        if data.get("kind") == "paper" and data.get("arxiv_id") == arxiv_id:
            return node_id
    return None


def _deslug_concept_key(slug: str) -> str:
    """
    Convert a URL slug like 'test-concept' back into a human label 'test concept'.
    """
    return slug.replace("-", " ")


def _slugify(label: str) -> str:
    """
    Very simple slugifier used for 'key' fields:
        'Test Concept' -> 'test-concept'
    """
    return label.strip().lower().replace(" ", "-")


def _find_concept_node(G: nx.MultiDiGraph, concept_slug: str) -> Optional[str]:
    """
    Locate a concept node given a slug from the URL path.

    We try the following, in order:

    1. Direct node id patterns:
        - 'concept:{slug}'          (e.g. 'concept:test-concept')
        - 'concept::{label}'        (e.g. 'concept::test concept')
        - '{slug}'                  (e.g. 'test-concept')
        - '{label}'                 (e.g. 'test concept')
    2. Attribute-based match on:
        - 'label'
        - 'base_name'
        - 'concept_key'
        - 'slug'
    """
    label = _deslug_concept_key(concept_slug)

    # 1) Try common node id patterns used in tests and pipeline
    candidate_ids = [
        f"concept:{concept_slug}",
        f"concept::{label}",
        concept_slug,
        label,
    ]
    for nid in candidate_ids:
        if nid in G:
            data = G.nodes[nid]
            if data.get("kind") == "concept":
                return nid

    # 2) Fallback to attribute-based search
    for node_id, data in G.nodes(data=True):
        if data.get("kind") != "concept":
            continue

        if concept_slug == data.get("slug"):
            return node_id

        if label in {
            data.get("label"),
            data.get("base_name"),
            data.get("concept_key"),
        }:
            return node_id

    return None


@lru_cache(maxsize=1)
def _cached_search_nodes() -> List[Dict[str, Any]]:
    """
    Build a simple in-memory search index over graph nodes.

    NOTE: Tests reach in and call `_cached_search_nodes.cache_clear()`, so the
    name and the fact that it's LRU-cached are intentional API.

    Each entry in the returned list is:
        {
            "id": <node_id>,          # actual graph node id
            "kind": <"paper"/"concept"/...>,
            "label": <best human label>,
            "text": <searchable text blob>,
        }
    """
    G = get_graph()

    index: List[Dict[str, Any]] = []

    for node_id, data in G.nodes(data=True):
        kind = data.get("kind")

        # Heuristic for a human-readable label
        label = (
            data.get("title")
            or data.get("label")
            or data.get("base_name")
            or data.get("concept_key")
            or data.get("arxiv_id")
            or str(node_id)
        )

        # Basic full-text content
        corpus_bits = [
            str(node_id),
            kind or "",
            str(label),
            str(data.get("abstract", "")),
            str(data.get("arxiv_id", "")),
        ]
        text = " ".join(corpus_bits).lower()

        index.append(
            {
                "id": node_id,     # IMPORTANT: real graph node id, no munging
                "kind": kind,
                "label": label,
                "text": text,
            }
        )

    return index


# -----------------------------------------------------------------------------
# Response models
# -----------------------------------------------------------------------------

class ConceptSummaryModel(BaseModel):
    id: str
    # "key" is the slug used in URLs, tests expect this on concept objects
    key: str
    label: str
    relation: Optional[str] = None
    weight: Optional[float] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class PaperModel(BaseModel):
    id: str
    arxiv_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class PaperWithConceptsResponse(BaseModel):
    paper: PaperModel
    concepts: List[ConceptSummaryModel]


class ConceptWithPapersResponse(BaseModel):
    concept: ConceptSummaryModel
    papers: List[PaperModel]


class SearchNodeResult(BaseModel):
    # tests expect "node_id" field in hits
    node_id: str
    kind: Optional[str] = None
    label: str


class SearchNodesResponse(BaseModel):
    query: str
    hits: List[SearchNodeResult]


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/papers/{arxiv_id}", response_model=PaperWithConceptsResponse)
def get_paper_with_concepts(arxiv_id: str) -> PaperWithConceptsResponse:
    """
    Retrieve a paper by arxiv_id plus its concept neighbours.

    - Finds the paper node by `kind="paper"` and `arxiv_id` attribute.
    - Follows outgoing HAS_CONCEPT edges to concept nodes.
    """
    G = get_graph()

    node_id = _find_paper_node(G, arxiv_id)
    if node_id is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    data = G.nodes[node_id]

    # Core fields
    paper = PaperModel(
        id=str(node_id),
        arxiv_id=arxiv_id,
        title=data.get("title"),
        abstract=data.get("abstract"),
        attributes={
            k: v
            for k, v in data.items()
            if k not in {"kind", "arxiv_id", "title", "abstract"}
        },
    )

    concepts: List[ConceptSummaryModel] = []
    # Paper -> Concept edges with relation "HAS_CONCEPT"
    for _, concept_id, edge_data in G.out_edges(node_id, data=True):
        c_data = G.nodes[concept_id]
        if c_data.get("kind") != "concept":
            continue

        relation = edge_data.get("relation")
        if relation and relation != "HAS_CONCEPT":
            continue

        label = (
            c_data.get("label")
            or c_data.get("base_name")
            or c_data.get("concept_key")
            or str(concept_id)
        )
        key = _slugify(str(label))

        concepts.append(
            ConceptSummaryModel(
                id=str(concept_id),
                key=key,
                label=str(label),
                relation=relation,
                weight=edge_data.get("weight"),
                attributes={
                    k: v
                    for k, v in c_data.items()
                    if k not in {"kind", "label", "base_name", "concept_key"}
                },
            )
        )

    return PaperWithConceptsResponse(paper=paper, concepts=concepts)


@app.get("/concepts/{concept_slug}", response_model=ConceptWithPapersResponse)
def get_concept_with_papers(concept_slug: str) -> ConceptWithPapersResponse:
    """
    Retrieve a concept (by slugified label) and all papers that reference it.

    - URL uses a slug: 'test-concept'
    - We deslug to 'test concept' and match against node ids or concept node attributes.
    - Papers are any `kind="paper"` nodes with a HAS_CONCEPT edge to this concept.
    """
    G = get_graph()

    node_id = _find_concept_node(G, concept_slug)
    if node_id is None:
        raise HTTPException(status_code=404, detail="Concept not found")

    c_data = G.nodes[node_id]
    label = (
        c_data.get("label")
        or c_data.get("base_name")
        or c_data.get("concept_key")
        or _deslug_concept_key(concept_slug)
    )

    concept = ConceptSummaryModel(
        id=str(node_id),
        key=concept_slug,  # tests expect the same slug back as "key"
        label=str(label),
        relation=None,
        weight=None,
        attributes={
            k: v
            for k, v in c_data.items()
            if k not in {"kind", "label", "base_name", "concept_key"}
        },
    )

    papers: List[PaperModel] = []

    # Expect edges Paper -> Concept with relation HAS_CONCEPT
    for paper_id in G.predecessors(node_id):
        p_data = G.nodes[paper_id]
        if p_data.get("kind") != "paper":
            continue

        papers.append(
            PaperModel(
                id=str(paper_id),
                arxiv_id=p_data.get("arxiv_id", str(paper_id)),
                title=p_data.get("title"),
                abstract=p_data.get("abstract"),
                attributes={
                    k: v
                    for k, v in p_data.items()
                    if k not in {"kind", "arxiv_id", "title", "abstract"}
                },
            )
        )

    return ConceptWithPapersResponse(concept=concept, papers=papers)


@app.get("/search/nodes", response_model=SearchNodesResponse)
def search_nodes(
    q: str = Query(..., min_length=1),
    kind: Optional[str] = Query(None, description="Optional filter by node kind."),
) -> SearchNodesResponse:
    """
    Simple substring search over node labels / metadata.

    IMPORTANT: We always use the real node id from the graph; we never
    transform it (no slugging, no prefix changes), which avoids KeyError
    when looking back into the graph.
    """
    query = q.strip().lower()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    index = _cached_search_nodes()

    hits: List[SearchNodeResult] = []
    for entry in index:
        if kind and entry["kind"] != kind:
            continue
        if query in entry["text"]:
            hits.append(
                SearchNodeResult(
                    node_id=str(entry["id"]),
                    kind=entry["kind"],
                    label=str(entry["label"]),
                )
            )

    return SearchNodesResponse(query=q, hits=hits)
