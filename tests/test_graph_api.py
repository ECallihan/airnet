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
