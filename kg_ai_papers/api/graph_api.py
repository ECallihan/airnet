from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from kg_ai_papers.graph.io import load_graph, save_graph
from kg_ai_papers.ingest.pipeline import ingest_arxiv_paper, IngestedPaperResult
from kg_ai_papers.graph.builder import update_graph_with_ingested_paper

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

class PaperModel(BaseModel):
    id: str
    arxiv_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ConceptSummaryModel(BaseModel):
    id: str
    key: str
    label: str
    relation: Optional[str] = None
    weight: Optional[float] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class NeighborModel(BaseModel):
    id: str
    kind: Optional[str] = None
    label: Optional[str] = None
    relation: Optional[str] = None
    direction: str = Field(
        ...,
        description="Direction of the edge relative to the paper: 'out' or 'in'.",
    )
    weight: Optional[float] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class PaperWithConceptsResponse(BaseModel):
    paper: PaperModel
    concepts: List[ConceptSummaryModel]
    neighbors: List[NeighborModel] = Field(
        default_factory=list,
        description=(
            "Non-concept neighbour nodes (typically other papers) directly "
            "connected to this paper."
        ),
    )



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


class ConceptStats(BaseModel):
    id: str
    key: str
    label: str
    degree: int


class GraphStatsResponse(BaseModel):
    num_nodes: int = Field(
        ...,
        description="Total number of nodes in the graph.",
    )
    num_edges: int = Field(
        ...,
        description="Total number of edges in the graph.",
    )
    num_papers: int = Field(
        ...,
        description="Number of paper nodes (kind='paper').",
    )
    num_concepts: int = Field(
        ...,
        description="Number of concept nodes (kind='concept').",
    )
    top_concepts: List[ConceptStats] = Field(
        default_factory=list,
        description="Top concepts ranked by degree.",
    )

class IngestedPaperView(BaseModel):
    """Minimal view of an ingested paper for API responses."""
    arxiv_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    num_concepts: int = Field(
        ...,
        description="Number of aggregated concepts extracted for this paper.",
    )
    num_references: int = Field(
        ...,
        description="Number of references (arXiv ids) this paper cites.",
    )


class IngestFailure(BaseModel):
    arxiv_id: str
    error: str


class BatchIngestRequest(BaseModel):
    """Request body for batch arXiv ingestion."""
    ids: List[str] = Field(
        ...,
        description="List of arXiv IDs to ingest.",
    )
    work_dir: Optional[str] = Field(
        None,
        description="Optional override for ingestion working directory.",
    )
    use_cache: bool = Field(
        True,
        description="If true, use ingestion cache when available.",
    )
    force_reingest: bool = Field(
        False,
        description="If true, ignore existing cache and recompute ingestion.",
    )


class BatchIngestResponse(BaseModel):
    """Response for batch arXiv ingestion."""
    requested_ids: List[str]
    ingested: List[IngestedPaperView] = Field(
        default_factory=list,
        description="Successfully ingested papers.",
    )
    failed: List[IngestFailure] = Field(
        default_factory=list,
        description="Papers that failed to ingest.",
    )
    graph_num_nodes: int = Field(
        ...,
        description="Total number of nodes in the updated graph.",
    )
    graph_num_edges: int = Field(
        ...,
        description="Total number of edges in the updated graph.",
    )

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.post("/ingest/arxiv", response_model=BatchIngestResponse)
def ingest_arxiv_batch(request: BatchIngestRequest) -> BatchIngestResponse:
    """Ingest a batch of arXiv papers and update the on-disk graph."""
    # Normalize and de-duplicate ids while preserving order
    raw_ids = [id_.strip() for id_ in (request.ids or []) if id_.strip()]
    seen = set()
    ids: List[str] = []
    for arxiv_id in raw_ids:
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            ids.append(arxiv_id)

    if not ids:
        raise HTTPException(status_code=400, detail="No arXiv IDs provided.")

    work_dir = Path(request.work_dir) if request.work_dir else Path("data/ingest")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load existing graph if present, otherwise start a new one
    try:
        G = get_graph()
    except FileNotFoundError:
        G = nx.MultiDiGraph()

    ingested_views: List[IngestedPaperView] = []
    failures: List[IngestFailure] = []

    for arxiv_id in ids:
        try:
            result = ingest_arxiv_paper(
                arxiv_id=arxiv_id,
                work_dir=work_dir,
                use_cache=request.use_cache,
                force_reingest=request.force_reingest,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                IngestFailure(
                    arxiv_id=arxiv_id,
                    error=str(exc),
                )
            )
            continue

        # Update in-memory graph with this paper
        update_graph_with_ingested_paper(G, result)

        # Build a small view for the response
        num_concepts = len(getattr(result, "concept_summaries", {}) or {})
        num_refs = len(getattr(result, "references", []) or [])

        ingested_views.append(
            IngestedPaperView(
                arxiv_id=result.paper.arxiv_id,
                title=result.paper.title,
                abstract=result.paper.abstract,
                num_concepts=num_concepts,
                num_references=num_refs,
            )
        )

    if not ingested_views:
        # Nothing succeeded; signal this to the client
        raise HTTPException(
            status_code=400,
            detail="No papers were successfully ingested.",
        )

    # Persist the updated graph to disk
    graph_path = _get_graph_path()
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(G, graph_path)

    # Clear search cache so /search/nodes sees new content
    _cached_search_nodes.cache_clear()

    return BatchIngestResponse(
        requested_ids=ids,
        ingested=ingested_views,
        failed=failures,
        graph_num_nodes=G.number_of_nodes(),
        graph_num_edges=G.number_of_edges(),
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/papers/{arxiv_id}", response_model=PaperWithConceptsResponse)
def get_paper_with_concepts(arxiv_id: str) -> PaperWithConceptsResponse:
    """
    Retrieve a paper by arxiv_id plus its concept neighbours and immediate graph neighbours.

    Response:
    - paper: core metadata for the paper
    - concepts: concept neighbours via outgoing edges
    - neighbors: non-concept neighbours (e.g. other papers) via in/out edges
    """
    G = get_graph()

    node_id = _find_paper_node(G, arxiv_id)
    if node_id is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    data = G.nodes[node_id]

    paper_attrs = {
        k: v
        for k, v in data.items()
        if k
        not in {
            "kind",
            "label",
            "base_name",
            "concept_key",
            "slug",
            "arxiv_id",
            "title",
            "abstract",
        }
    }

    paper = PaperModel(
        id=str(node_id),
        arxiv_id=str(data.get("arxiv_id", arxiv_id)),
        title=data.get("title"),
        abstract=data.get("abstract"),
        attributes=paper_attrs,
    )

    # -------- concepts: outgoing edges to concept nodes --------
    concepts: List[ConceptSummaryModel] = []
    for _, nbr, edge_data in G.out_edges(node_id, data=True):
        nbr_data = G.nodes[nbr]
        if nbr_data.get("kind") != "concept":
            continue

        key = (
            nbr_data.get("concept_key")
            or nbr_data.get("slug")
            or str(nbr).split("concept:")[-1]
        )
        label = nbr_data.get("label") or key
        relation = edge_data.get("relation") or edge_data.get("type")

        concept_attrs = {
            k: v
            for k, v in nbr_data.items()
            if k not in {"kind", "label", "base_name", "concept_key", "slug"}
        }

        concepts.append(
            ConceptSummaryModel(
                id=str(nbr),
                key=str(key),
                label=str(label),
                relation=relation,
                weight=edge_data.get("weight"),
                attributes=concept_attrs,
            )
        )

    # -------- neighbors: both directions, non-concept only --------
    neighbors: List[NeighborModel] = []

    def _add_neighbor(src: Any, dst: Any, edge_data: Dict[str, Any], direction: str):
        if dst == node_id:
            return

        n_data = G.nodes[dst]

        # Skip concept nodes; these are reported via 'concepts'
        if n_data.get("kind") == "concept":
            return

        rel = edge_data.get("relation") or edge_data.get("type")
        label = n_data.get("label") or n_data.get("title") or str(dst)
        attrs = {
            k: v
            for k, v in n_data.items()
            if k not in {"kind", "label", "base_name", "concept_key", "slug"}
        }

        neighbors.append(
            NeighborModel(
                id=str(dst),
                kind=n_data.get("kind"),
                label=str(label),
                relation=rel,
                direction=direction,
                weight=edge_data.get("weight"),
                attributes=attrs,
            )
        )

    # Outgoing neighbours
    for _, nbr, edge_data in G.out_edges(node_id, data=True):
        _add_neighbor(node_id, nbr, edge_data, direction="out")

    # Incoming neighbours
    for nbr, _, edge_data in G.in_edges(node_id, data=True):
        _add_neighbor(nbr, node_id, edge_data, direction="in")

    return PaperWithConceptsResponse(
        paper=paper,
        concepts=concepts,
        neighbors=neighbors,
    )


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
    kind: Optional[str] = Query(
        None,
        description="Optional filter by node kind.",
    ),
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

@app.get("/graph/stats", response_model=GraphStatsResponse)
def graph_stats(
    top_k: int = Query(
        10,
        ge=1,
        le=100,
        description="Number of top concepts to return, ranked by graph degree.",
    )
) -> GraphStatsResponse:
    """
    Return high-level statistics about the current graph.

    This includes node/edge counts, paper/concept counts, and the top-k
    concept nodes ranked by their total graph degree.
    """
    G = get_graph()

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    num_papers = 0
    num_concepts = 0
    concept_degrees: List[Dict[str, Any]] = []

    for node_id, data in G.nodes(data=True):
        kind = data.get("kind")
        if kind == "paper":
            num_papers += 1
        elif kind == "concept":
            num_concepts += 1
            degree = int(G.degree(node_id))
            key = (
                data.get("concept_key")
                or data.get("slug")
                or str(node_id).split("concept:")[-1]
            )
            label = data.get("label") or key
            concept_degrees.append(
                {
                    "id": str(node_id),
                    "key": str(key),
                    "label": str(label),
                    "degree": degree,
                }
            )

    concept_degrees.sort(key=lambda x: x["degree"], reverse=True)
    top_items = concept_degrees[:top_k]

    top_concepts = [
        ConceptStats(
            id=item["id"],
            key=item["key"],
            label=item["label"],
            degree=item["degree"],
        )
        for item in top_items
    ]

    return GraphStatsResponse(
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_papers=num_papers,
        num_concepts=num_concepts,
        top_concepts=top_concepts,
    )
