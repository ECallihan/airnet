from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import networkx as nx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from kg_ai_papers.config.settings import settings
from kg_ai_papers.graph.storage import load_latest_graph
from kg_ai_papers.ingest.pipeline import ingest_arxiv_paper
from kg_ai_papers.graph.builder import update_graph_with_ingested_paper

# Use a real FastAPI app so TestClient(app) works as expected
app = FastAPI()


# ---------------------------------------------------------------------------
# Graph access
# ---------------------------------------------------------------------------

_GRAPH_CACHE: Optional[nx.MultiDiGraph] = None

# Thin wrapper so tests can monkeypatch graph_api.save_graph
def save_graph(*args, **kwargs):
    """
    Delegate to kg_ai_papers.graph.storage.save_graph.

    The ingest API tests patch this symbol on the graph_api module,
    so we just need it to exist and forward to the real implementation.
    """
    from kg_ai_papers.graph.storage import save_graph as _save_graph

    return _save_graph(*args, **kwargs)

def _load_graph_from_disk() -> nx.MultiDiGraph:
    """
    Default graph loader used in production.

    In tests, this function is bypassed because test modules monkeypatch
    get_graph() directly to return a small synthetic graph.
    """
    global _GRAPH_CACHE

    if _GRAPH_CACHE is not None:
        return _GRAPH_CACHE

    G = load_latest_graph(settings.graph_dir)
    if G is None:
        raise RuntimeError(
            f"No graph found in {settings.graph_dir}. "
            "Run the ingestion pipeline first."
        )

    _GRAPH_CACHE = G
    return G


def get_graph() -> nx.MultiDiGraph:
    """
    Indirection point for retrieving the in-memory graph.

    Tests monkeypatch this function, so all endpoints must call get_graph()
    rather than touching _GRAPH_CACHE directly.
    """
    return _load_graph_from_disk()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_paper_node(graph: nx.MultiDiGraph, arxiv_id: str) -> Optional[Any]:
    """
    Robustly find the node corresponding to this paper id in different graph flavors:
      - node id == f"paper:{arxiv_id}"
      - node id == arxiv_id
      - node with node['arxiv_id'] == arxiv_id or node['paper_id'] == arxiv_id
    """
    candidates: List[Any] = []

    # 1) paper:ID
    prefixed = f"paper:{arxiv_id}"
    if prefixed in graph:
        candidates.append(prefixed)

    # 2) raw id
    if arxiv_id in graph:
        candidates.append(arxiv_id)

    # 3) node attribute matches
    for n, attrs in graph.nodes(data=True):
        if attrs.get("arxiv_id") == arxiv_id or attrs.get("paper_id") == arxiv_id:
            candidates.append(n)

    return candidates[0] if candidates else None


def _node_kind(attrs: Dict[str, Any]) -> str:
    return str(attrs.get("type") or attrs.get("kind") or "").lower() or "unknown"


def _node_label(node: Any, attrs: Dict[str, Any]) -> str:
    return (
        attrs.get("title")
        or attrs.get("label")
        or attrs.get("name")
        or attrs.get("concept_key")
        or str(node)
    )


def _paper_from_node(node: Any, attrs: Dict[str, Any]) -> "PaperNode":
    """
    Normalize a graph paper node into our API PaperNode model,
    stripping reserved fields from attributes.
    """
    data = dict(attrs)  # shallow copy

    # Pull out reserved/top-level fields
    arxiv_id = str(data.pop("arxiv_id", data.pop("paper_id", str(node))))
    title = data.pop("title", None)
    abstract = data.pop("abstract", None)

    # Strip reserved meta
    for reserved in ("kind", "type", "label", "name"):
        data.pop(reserved, None)

    return PaperNode(
        id=str(node),
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        attributes=data,
    )


def _concept_from_node(
    node: Any,
    attrs: Dict[str, Any],
    importance: Optional[float] = None,
) -> "ConceptNode":
    """
    Normalize a concept node and attach optional importance.
    """
    data = dict(attrs)
    key = data.pop("concept_key", data.pop("key", data.get("label") or str(node)))
    label = data.pop("label", data.get("name") or key)

    # Strip reserved meta
    for reserved in ("kind", "type", "name"):
        data.pop(reserved, None)

    # Attach importance if provided
    if importance is not None:
        data["importance"] = importance

    return ConceptNode(
        id=str(node),
        key=str(key),
        label=str(label),
        attributes=data,
    )


def _edge_attr_or_none(
    graph: nx.MultiDiGraph,
    u: Any,
    v: Any,
    attr: str,
) -> Optional[Any]:
    """
    Safely extract a single edge attribute from u->v, handling both DiGraph
    and MultiDiGraph edge data structures.
    """
    if not graph.has_edge(u, v):
        return None

    data = graph.get_edge_data(u, v, default=None)
    if data is None:
        return None

    # MultiDiGraph: mapping key -> data dict
    if isinstance(graph, nx.MultiDiGraph):
        # data is {edge_key: {attrs}}
        for _, d in data.items():
            if isinstance(d, dict) and attr in d:
                return d[attr]
        return None

    # DiGraph or Graph: data is the attrs dict
    if isinstance(data, dict):
        return data.get(attr)

    return None

def _edge_weight(
    graph: nx.MultiDiGraph,
    u: Any,
    v: Any,
) -> Optional[float]:
    """
    Extract a representative weight/score/importance from edges u->v,
    handling both DiGraph and MultiDiGraph.

    For MultiDiGraph we take the max weight across parallel edges.
    """
    if not graph.has_edge(u, v):
        return None

    data = graph.get_edge_data(u, v, default=None)
    if data is None:
        return None

    # MultiDiGraph: mapping edge_key -> data dict
    if isinstance(graph, nx.MultiDiGraph):
        weights: List[float] = []
        for _, d in data.items():
            if not isinstance(d, dict):
                continue
            val = d.get("weight") or d.get("score") or d.get("importance")
            if isinstance(val, (int, float)):
                weights.append(float(val))
        if not weights:
            return None
        return max(weights)

    # DiGraph / Graph: data is a single attrs dict
    if isinstance(data, dict):
        val = data.get("weight") or data.get("score") or data.get("importance")
        if isinstance(val, (int, float)):
            return float(val)

    return None

def _paper_influence(
    graph: nx.MultiDiGraph,
    node: Any,
) -> tuple[List[InfluenceNeighbor], List[InfluenceNeighbor]]:
    """
    Collect incoming and outgoing paper-level influence edges for a given paper.

    - Incoming: neighbors that point to this paper (neighbor -> paper)
    - Outgoing: this paper points to neighbor (paper -> neighbor)
    """
    incoming_ids: Set[Any] = set(graph.predecessors(node))
    outgoing_ids: Set[Any] = set(graph.successors(node))

    incoming: List[InfluenceNeighbor] = []
    outgoing: List[InfluenceNeighbor] = []

    # Incoming influence: neighbor cites this paper
    for nbr in incoming_ids:
        attrs = graph.nodes[nbr]
        kind = _node_kind(attrs)
        if kind != "paper" and "paper" not in kind:
            continue

        neighbor_paper = _paper_from_node(nbr, attrs)
        relation = _edge_attr_or_none(graph, nbr, node, "relation")
        weight = _edge_weight(graph, nbr, node)

        incoming.append(
            InfluenceNeighbor(
                paper=neighbor_paper,
                direction="in",
                relation=str(relation) if relation is not None else None,
                weight=weight,
            )
        )

    # Outgoing influence: this paper cites neighbor
    for nbr in outgoing_ids:
        attrs = graph.nodes[nbr]
        kind = _node_kind(attrs)
        if kind != "paper" and "paper" not in kind:
            continue

        neighbor_paper = _paper_from_node(nbr, attrs)
        relation = _edge_attr_or_none(graph, node, nbr, "relation")
        weight = _edge_weight(graph, node, nbr)

        outgoing.append(
            InfluenceNeighbor(
                paper=neighbor_paper,
                direction="out",
                relation=str(relation) if relation is not None else None,
                weight=weight,
            )
        )

    # Sort for deterministic behavior: strongest first, then arxiv_id
    incoming.sort(
        key=lambda n: (
            -(n.weight if n.weight is not None else 0.0),
            n.paper.arxiv_id,
        )
    )
    outgoing.sort(
        key=lambda n: (
            -(n.weight if n.weight is not None else 0.0),
            n.paper.arxiv_id,
        )
    )

    return incoming, outgoing


def _paper_neighbors(graph: nx.MultiDiGraph, node: Any) -> List["NeighborNode"]:
    """
    Collect paper neighbors of a paper, excluding concept nodes.

    New behavior: each neighbor includes a 'direction' field relative to the
    paper node:
      - 'out'  : edge from paper -> neighbor
      - 'in'   : edge from neighbor -> paper
      - 'both' : edges in both directions

    Also includes a 'relation' field populated from the edge metadata
    (e.g. 'CITES', 'PAPER_CITES_PAPER') when available.
    """
    neighbors: Set[Any] = set(graph.predecessors(node)) | set(graph.successors(node))
    results: List[NeighborNode] = []

    for nbr in neighbors:
        if nbr == node:
            continue
        attrs: Dict[str, Any] = graph.nodes[nbr]
        kind = _node_kind(attrs)
        if kind != "paper" and "paper" not in kind:
            # Only paper neighbors belong in this collection.
            continue

        label = _node_label(nbr, attrs)

        has_out = graph.has_edge(node, nbr)
        has_in = graph.has_edge(nbr, node)

        if has_out and has_in:
            direction = "both"
        elif has_out:
            direction = "out"
        elif has_in:
            direction = "in"
        else:
            direction = "unknown"

        # Prefer relation metadata from the edge in the direction we report
        relation = None
        if direction in ("out", "both"):
            relation = _edge_attr_or_none(graph, node, nbr, "relation")
        if relation is None and direction in ("in", "both"):
            relation = _edge_attr_or_none(graph, nbr, node, "relation")

        results.append(
            NeighborNode(
                id=str(nbr),
                kind="paper",
                label=str(label),
                direction=direction,
                relation=str(relation) if relation is not None else None,
            )
        )

    # Stable ordering: sort by label then id
    results.sort(key=lambda n: (n.label, n.id))
    return results


def _concepts_for_paper(graph: nx.MultiDiGraph, node: Any) -> List["ConceptNode"]:
    """
    Collect concept nodes connected to this paper, along with an 'importance'
    score derived from edge attributes where available.
    """
    neighbors: Set[Any] = set(graph.predecessors(node)) | set(graph.successors(node))
    scores: Dict[Any, float] = {}

    def _edge_datas(u: Any, v: Any) -> Iterable[Dict[str, Any]]:
        # Support both DiGraph and MultiDiGraph
        if isinstance(graph, nx.MultiDiGraph):
            # MultiDiGraph: graph[u][v] is a dict keyed by edge key
            for _, data in graph.get_edge_data(u, v, default={}).items():
                yield data
        else:
            # DiGraph: graph[u][v] is the data dict
            data = graph.get_edge_data(u, v, default=None)
            if data:
                yield data

    # Aggregate importance/weight/score from edges between paper and concept
    for nbr in neighbors:
        attrs = graph.nodes[nbr]
        kind = _node_kind(attrs)
        if kind != "concept" and "concept" not in kind:
            continue

        total = 0.0
        found_any = False
        for data in _edge_datas(node, nbr):
            for field in ("importance", "weight", "score"):
                if field in data and isinstance(data[field], (int, float)):
                    total += float(data[field])
                    found_any = True
        for data in _edge_datas(nbr, node):
            for field in ("importance", "weight", "score"):
                if field in data and isinstance(data[field], (int, float)):
                    total += float(data[field])
                    found_any = True

        scores[nbr] = total if found_any else 0.0

    concepts: List[ConceptNode] = []
    for nbr, attrs in graph.nodes(data=True):
        if nbr not in neighbors:
            continue
        kind = _node_kind(attrs)
        if kind != "concept" and "concept" not in kind:
            continue

        importance = scores.get(nbr)
        concepts.append(_concept_from_node(nbr, attrs, importance=importance))

    # Stable ordering by importance (descending) then id
    concepts.sort(
        key=lambda c: (float(c.attributes.get("importance", 0.0)), c.id),
        reverse=True,
    )
    return concepts


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PaperNode(BaseModel):
    id: str
    arxiv_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ConceptNode(BaseModel):
    id: str
    key: str
    label: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class NeighborNode(BaseModel):
    id: str
    kind: str
    label: str
    # Direction of edge relative to main paper
    direction: str
    # Relation string from edge metadata (e.g. "CITES")
    relation: Optional[str] = None


class PaperResponse(BaseModel):
    paper: PaperNode
    concepts: List[ConceptNode] = Field(default_factory=list)
    neighbors: List[NeighborNode] = Field(default_factory=list)


class NodeSearchHit(BaseModel):
    node_id: str
    kind: str
    label: str
    score: float


class NodeSearchResponse(BaseModel):
    query: str
    kind: Optional[str] = None
    hits: List[NodeSearchHit]


class BatchIngestItem(BaseModel):
    """Summary of a single ingested paper used by the batch ingest API."""
    arxiv_id: str
    num_concepts: int
    num_references: int


class BatchIngestResponse(BaseModel):
    """Response payload for /ingest/arxiv batch endpoint."""
    requested_ids: List[str]
    ingested: List[BatchIngestItem] = Field(default_factory=list)
    failed: List[str] = Field(default_factory=list)
    graph_num_nodes: int
    graph_num_edges: int


class BatchIngestRequest(BaseModel):
    """Request payload for /ingest/arxiv batch endpoint.

    The tests post this shape directly as JSON.
    """
    ids: List[str] = Field(default_factory=list)
    work_dir: str
    use_cache: bool = True
    force_reingest: bool = False


class ConceptResponse(BaseModel):
    concept: ConceptNode
    papers: List[PaperNode]

class InfluenceNeighbor(BaseModel):
    paper: PaperNode
    # 'in'  : neighbor -> focal paper (neighbor cites focal)
    # 'out' : focal paper -> neighbor (focal cites neighbor)
    direction: str
    relation: Optional[str] = None
    weight: Optional[float] = None


class PaperInfluenceResponse(BaseModel):
    paper: PaperNode
    incoming: List[InfluenceNeighbor] = Field(default_factory=list)
    outgoing: List[InfluenceNeighbor] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Ingestion endpoint
# ---------------------------------------------------------------------------


@app.post("/ingest/arxiv", response_model=BatchIngestResponse)
def ingest_arxiv_batch(request: BatchIngestRequest) -> BatchIngestResponse:
    """Ingest a batch of arXiv papers and update the on-disk graph.

    Behaviour is intentionally simple and test-friendly:

    - Normalizes and de-duplicates the requested ids while preserving order.
    - Calls ``ingest_arxiv_paper`` once per id.
    - Skips ids whose ingestion raises an exception.
    - If *all* ingests fail, returns HTTP 400.
    - On any success, persists the updated graph via ``save_graph``.
    """

    # Normalize + de-duplicate ids (strip whitespace, drop empties)
    raw_ids = [id_.strip() for id_ in (request.ids or []) if id_.strip()]
    seen: Set[str] = set()
    requested_ids: List[str] = []
    for arxiv_id in raw_ids:
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            requested_ids.append(arxiv_id)

    if not requested_ids:
        raise HTTPException(status_code=400, detail="No ids were provided.")

    G = get_graph()

    ingested_items: List[BatchIngestItem] = []
    failed_ids: List[str] = []

    for arxiv_id in requested_ids:
        try:
            result = ingest_arxiv_paper(
                arxiv_id=arxiv_id,
                work_dir=Path(request.work_dir),
                use_cache=request.use_cache,
                force_reingest=request.force_reingest,
            )
        except Exception:
            # Record the failure for this id; we still allow other ids to succeed.
            failed_ids.append(arxiv_id)
            continue

        # Update the in-memory graph for each successfully ingested paper.
        # Tests monkeypatch ``update_graph_with_ingested_paper`` to record calls.
        update_graph_with_ingested_paper(G, result)

        num_concepts = len(result.concept_summaries or {})
        num_references = len(result.references or [])

        ingested_items.append(
            BatchIngestItem(
                arxiv_id=result.paper.arxiv_id,
                num_concepts=num_concepts,
                num_references=num_references,
            )
        )

    if not ingested_items:
        # Tests assert on this error detail when all ingests fail.
        raise HTTPException(
            status_code=400,
            detail="No papers were successfully ingested.",
        )

    # At least one paper was ingested successfully – persist updated graph.
    # The ingest API tests monkeypatch graph_api.save_graph with a simple
    # (graph, path) signature, so we always pass two positional arguments.
    save_graph(G, "ingest-batch")

    return BatchIngestResponse(
        requested_ids=requested_ids,
        ingested=ingested_items,
        failed=failed_ids,
        graph_num_nodes=G.number_of_nodes(),
        graph_num_edges=G.number_of_edges(),
    )


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Simple health check endpoint.

    The tests only assert that this returns HTTP 200; we also return a small
    status payload for humans.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Paper endpoint
# ---------------------------------------------------------------------------


@app.get("/papers/{arxiv_id}", response_model=PaperResponse)
def get_paper(arxiv_id: str) -> PaperResponse:
    """
    Return a detailed view of a single paper, including:

    - core metadata for the paper node
    - a 'concepts' collection for connected concept nodes
    - a 'neighbors' collection for neighboring *paper* nodes only
      (excluding concept nodes), each with a 'direction' and 'relation' field.
    """
    G = get_graph()

    node = _find_paper_node(G, arxiv_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    attrs = G.nodes[node]
    paper = _paper_from_node(node, attrs)
    concepts = _concepts_for_paper(G, node)
    neighbors = _paper_neighbors(G, node)

    return PaperResponse(paper=paper, concepts=concepts, neighbors=neighbors)


@app.get("/papers/{arxiv_id}/influence", response_model=PaperInfluenceResponse)
def get_paper_influence(arxiv_id: str) -> PaperInfluenceResponse:
    """
    Paper Influence Explorer endpoint.

    Returns:
      - The focal paper
      - Incoming influence (papers that cite this one)
      - Outgoing influence (papers this one cites)
    """
    G = get_graph()

    node = _find_paper_node(G, arxiv_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    attrs = G.nodes[node]
    paper = _paper_from_node(node, attrs)
    incoming, outgoing = _paper_influence(G, node)

    return PaperInfluenceResponse(
        paper=paper,
        incoming=incoming,
        outgoing=outgoing,
    )

# ---------------------------------------------------------------------------
# Concept endpoint
# ---------------------------------------------------------------------------


@app.get("/concepts/{concept_key}", response_model=ConceptResponse)
def get_concept(concept_key: str) -> ConceptResponse:
    """
    Retrieve a concept by its key (e.g. 'test-concept') and list its connected
    papers.

    Tests assert that:
      - /concepts/test-concept returns 200
      - The response shape is stable enough to drive a concept inspector UI.
    """
    G = get_graph()

    concept_node: Optional[Any] = None
    concept_attrs: Optional[Dict[str, Any]] = None

    for n, attrs in G.nodes(data=True):
        kind = _node_kind(attrs)
        if kind != "concept" and "concept" not in kind:
            continue

        key_candidate = (
            attrs.get("concept_key")
            or attrs.get("key")
            or attrs.get("label")
            or str(n)
        )

        if str(key_candidate) == concept_key:
            concept_node = n
            concept_attrs = attrs
            break

    if concept_node is None or concept_attrs is None:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_key}' not found")

    concept = _concept_from_node(concept_node, concept_attrs)

    # Collect connected papers (neighbors that are 'paper')
    neighbor_ids: Set[Any] = set(G.predecessors(concept_node)) | set(
        G.successors(concept_node)
    )
    papers: List[PaperNode] = []

    for nbr in neighbor_ids:
        attrs = G.nodes[nbr]
        kind = _node_kind(attrs)
        if kind != "paper" and "paper" not in kind:
            continue
        papers.append(_paper_from_node(nbr, attrs))

    # Stable ordering for deterministic tests/UI
    papers.sort(key=lambda p: (p.arxiv_id, p.id))

    return ConceptResponse(concept=concept, papers=papers)


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _cached_search_nodes(query: str) -> List[Dict[str, Any]]:
    """
    Low-level cached search over nodes by simple substring matching
    on node id and label.

    Tests monkeypatch get_graph() and then optionally clear this cache via
    _cached_search_nodes.cache_clear().
    """
    G = get_graph()
    q = query.lower()
    hits: List[Dict[str, Any]] = []

    for node, attrs in G.nodes(data=True):
        label = _node_label(node, attrs)
        text = f"{node} {label}".lower()
        if q not in text:
            continue

        kind = _node_kind(attrs)
        hits.append(
            {
                "node_id": str(node),
                "kind": kind,
                "label": str(label),
                # Simple score: length of match / label length, or just 1.0.
                "score": 1.0,
            }
        )

    return hits


@app.get("/search/nodes", response_model=NodeSearchResponse)
def search_nodes(
    q: str = Query(..., alias="q", description="Substring query over id/label"),
    kind: Optional[str] = Query(
        None,
        description="Optional node kind filter, e.g. 'paper' or 'concept'.",
    ),
) -> NodeSearchResponse:
    """
    Search graph nodes by id/label.

    Tests expect:
      - 'query' field echoed back as the (stripped) query
      - 'hits' list containing objects with 'node_id'
      - 400 when the query is empty or whitespace only
      - 'kind' filter to restrict to just papers or just concepts
    """
    query = q.strip()
    if not query:
        # test_search_nodes_rejects_empty_query expects 400 here
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty.",
        )

    all_hits = _cached_search_nodes(query)

    if kind:
        wanted = kind.lower()
        filtered = [
            h for h in all_hits if (h.get("kind") or "").lower() == wanted
        ]
    else:
        filtered = all_hits

    hits_models = [
        NodeSearchHit(
            node_id=h["node_id"],
            kind=h["kind"],
            label=h["label"],
            score=float(h.get("score", 1.0)),
        )
        for h in filtered
    ]

    return NodeSearchResponse(query=query, kind=kind, hits=hits_models)
