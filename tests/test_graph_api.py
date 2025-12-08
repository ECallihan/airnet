import networkx as nx
from fastapi.testclient import TestClient

from kg_ai_papers.api.graph_api import (
    app,
)


def _build_test_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # Paper node
    paper_id = "paper:1234.5678"
    G.add_node(
        paper_id,
        kind="paper",
        arxiv_id="1234.5678",
        title="Test Paper",
        abstract="Test Abstract",
        some_meta="foo",
    )

    # Concept nodes
    c1_id = "concept:test-concept"
    c2_id = "concept:other-concept"

    G.add_node(
        c1_id,
        kind="concept",
        concept_key="test-concept",
        label="Test Concept",
        importance=0.9,
    )
    G.add_node(
        c2_id,
        kind="concept",
        concept_key="other-concept",
        label="Other Concept",
        importance=0.3,
    )

    # Edges: paper -> concepts
    G.add_edge(paper_id, c1_id, key="HAS_CONCEPT", relation="HAS_CONCEPT", weight=0.9)
    G.add_edge(paper_id, c2_id, key="HAS_CONCEPT", relation="HAS_CONCEPT", weight=0.3)

    return G


def test_health_endpoint(monkeypatch):
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_get_paper_with_concepts(monkeypatch):
    from kg_ai_papers.api import graph_api

    G = _build_test_graph()

    # Monkeypatch get_graph and clear any caches that depend on it
    monkeypatch.setattr(graph_api, "get_graph", lambda: G)
    if hasattr(graph_api, "_cached_search_nodes"):
        graph_api._cached_search_nodes.cache_clear()  # type: ignore[attr-defined]

    client = TestClient(app)

    resp = client.get("/papers/1234.5678")
    assert resp.status_code == 200
    data = resp.json()

    assert data["paper"]["arxiv_id"] == "1234.5678"
    assert data["paper"]["title"] == "Test Paper"
    # attributes should include "some_meta" but not the reserved fields
    assert data["paper"]["attributes"].get("some_meta") == "foo"
    assert "kind" not in data["paper"]["attributes"]

    concepts = data["concepts"]
    assert len(concepts) == 2
    keys = {c["key"] for c in concepts}
    assert "test-concept" in keys
    assert "other-concept" in keys


def test_get_paper_not_found(monkeypatch):
    from kg_ai_papers.api import graph_api

    G = _build_test_graph()
    monkeypatch.setattr(graph_api, "get_graph", lambda: G)

    client = TestClient(app)

    resp = client.get("/papers/9999.00000")
    assert resp.status_code == 404


def test_get_concept_with_papers(monkeypatch):
    from kg_ai_papers.api import graph_api

    G = _build_test_graph()
    monkeypatch.setattr(graph_api, "get_graph", lambda: G)
    if hasattr(graph_api, "_cached_search_nodes"):
        graph_api._cached_search_nodes.cache_clear()  # type: ignore[attr-defined]

    client = TestClient(app)

    resp = client.get("/concepts/test-concept")
    assert resp.status_code == 200
    data = resp.json()

    assert data["concept"]["key"] == "test-concept"
    papers = data["papers"]
    assert len(papers) == 1
    assert papers[0]["arxiv_id"] == "1234.5678"


def test_search_nodes(monkeypatch):
    from kg_ai_papers.api import graph_api

    G = _build_test_graph()
    monkeypatch.setattr(graph_api, "get_graph", lambda: G)
    if hasattr(graph_api, "_cached_search_nodes"):
        graph_api._cached_search_nodes.cache_clear()  # type: ignore[attr-defined]

    client = TestClient(app)

    resp = client.get("/search/nodes", params={"q": "test"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == "test"
    hits = data["hits"]
    assert len(hits) >= 1

    node_ids = {hit["node_id"] for hit in hits}
    # Both the paper and "test-concept" should be plausible hits
    assert "paper:1234.5678" in node_ids
    assert "concept:test-concept" in node_ids

def test_get_paper_includes_neighbors_and_excludes_concepts(monkeypatch):
    """
    Ensure GET /papers/{arxiv_id} includes a 'neighbors' collection that:

    - Exists and is a list
    - Contains paper neighbors when present
    - Does NOT include concept nodes (those are in 'concepts')
    """
    from kg_ai_papers.api import graph_api

    G = _build_test_graph()

    # Add a neighbor paper and connect it to our main paper
    main_paper_id = "paper:1234.5678"
    neighbor_paper_id = "paper:9999.00000"

    G.add_node(
        neighbor_paper_id,
        kind="paper",
        arxiv_id="9999.00000",
        title="Neighbor Paper",
    )
    G.add_edge(
        main_paper_id,
        neighbor_paper_id,
        relation="CITES",
        weight=0.5,
    )

    # Hook our test graph into the API layer and clear any caches
    monkeypatch.setattr(graph_api, "get_graph", lambda: G)
    if hasattr(graph_api, "_cached_search_nodes"):
        graph_api._cached_search_nodes.cache_clear()  # type: ignore[attr-defined]

    client = TestClient(app)

    resp = client.get("/papers/1234.5678")
    assert resp.status_code == 200
    data = resp.json()

    # New neighbors field should be present
    assert "neighbors" in data
    neighbors = data["neighbors"]
    assert isinstance(neighbors, list)

    neighbor_ids = {n["id"] for n in neighbors}

    # Our extra paper neighbor should be present
    assert neighbor_paper_id in neighbor_ids

    # Concept nodes should *not* appear in neighbors
    assert "concept:test-concept" not in neighbor_ids
    assert "concept:other-concept" not in neighbor_ids

    # Sanity-check one neighbor entry
    neighbor_entry = next(n for n in neighbors if n["id"] == neighbor_paper_id)
    assert neighbor_entry["kind"] == "paper"
    # Direction is relative to the main paper node
    assert neighbor_entry["direction"] == "out"
    # Relation should come from the edge metadata
    assert neighbor_entry["relation"] in ("CITES", "PAPER_CITES_PAPER")


def test_search_nodes_kind_filter(monkeypatch):
    """
    Ensure /search/nodes supports filtering by node kind, so callers can
    ask specifically for papers or concepts.
    """
    from kg_ai_papers.api import graph_api

    G = _build_test_graph()
    monkeypatch.setattr(graph_api, "get_graph", lambda: G)
    if hasattr(graph_api, "_cached_search_nodes"):
        graph_api._cached_search_nodes.cache_clear()  # type: ignore[attr-defined]

    client = TestClient(app)

    # Only papers
    resp_papers = client.get("/search/nodes", params={"q": "test", "kind": "paper"})
    assert resp_papers.status_code == 200
    data_papers = resp_papers.json()
    ids_papers = {hit["node_id"] for hit in data_papers["hits"]}

    assert "paper:1234.5678" in ids_papers
    # Concepts should be filtered out here
    assert "concept:test-concept" not in ids_papers
    assert "concept:other-concept" not in ids_papers

    # Only concepts
    resp_concepts = client.get("/search/nodes", params={"q": "test", "kind": "concept"})
    assert resp_concepts.status_code == 200
    data_concepts = resp_concepts.json()
    ids_concepts = {hit["node_id"] for hit in data_concepts["hits"]}

    # At least one concept present
    assert "concept:test-concept" in ids_concepts
    # Paper should be filtered out
    assert "paper:1234.5678" not in ids_concepts



def test_search_nodes_rejects_empty_query(monkeypatch):
    """
    Ensure /search/nodes returns 400 for empty or whitespace-only queries,
    rather than silently returning everything.
    """
    from kg_ai_papers.api import graph_api

    G = _build_test_graph()
    monkeypatch.setattr(graph_api, "get_graph", lambda: G)
    if hasattr(graph_api, "_cached_search_nodes"):
        graph_api._cached_search_nodes.cache_clear()  # type: ignore[attr-defined]

    client = TestClient(app)

    resp = client.get("/search/nodes", params={"q": "   "})
    assert resp.status_code == 400

    detail = resp.json().get("detail", "")
    assert "empty" in detail.lower()
