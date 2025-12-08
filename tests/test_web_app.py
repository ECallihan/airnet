# tests/test_web_app.py

from fastapi.testclient import TestClient
import networkx as nx

from kg_ai_papers.web.app import app
from kg_ai_papers.graph.schema import NodeType, EdgeType


def make_mock_graph():
    G = nx.MultiDiGraph()
    G.add_node(
        "paper:p1",
        type=NodeType.PAPER.value,
        arxiv_id="p1",
        title="Mock Paper",
        abstract="Mock abstract.",
    )
    G.add_node(
        "concept:c1",
        type=NodeType.CONCEPT.value,
        label="mock concept",
    )
    G.add_edge(
        "paper:p1",
        "concept:c1",
        type=EdgeType.PAPER_HAS_CONCEPT.value,
        weight=1.0,
    )
    return G


def test_health_and_paper_endpoints(monkeypatch):
    # Override the app's graph state
    mock_graph = make_mock_graph()
    app.state.graph = mock_graph

    client = TestClient(app)

    # Health endpoint
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # List papers
    r = client.get("/papers")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert data[0]["arxiv_id"] == "p1"

    # Paper detail
    r = client.get("/papers/p1")
    assert r.status_code == 200
    detail = r.json()
    assert detail["paper"]["arxiv_id"] == "p1"
    assert len(detail["concepts"]) == 1
