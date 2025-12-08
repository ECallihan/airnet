# tests/test_web_read_endpoints.py

from typing import Dict, Any

from starlette.testclient import TestClient

from kg_ai_papers.web.app import app
import kg_ai_papers.web.app as web_app_module

# Reuse helpers from the ingest tests for consistency
from test_web_ingest import make_mock_graph, _make_dummy_influence_view


def test_list_papers_returns_summaries():
    """
    Ensure GET /papers returns a list of PaperSummary-like objects
    derived from graph nodes that have an 'arxiv_id' attribute.
    """
    client = TestClient(app)

    # Start from the same mock graph used in ingest tests
    G = make_mock_graph()

    # Add a couple of explicit paper nodes so the behavior is deterministic
    G.add_node(
        "paper:1111.11111",
        arxiv_id="1111.11111",
        title="Test Paper One",
        abstract="Abstract one",
    )
    G.add_node(
        "paper:2222.22222",
        arxiv_id="2222.22222",
        title="Test Paper Two",
        abstract="Abstract two",
    )

    app.state.graph = G

    response = client.get("/papers")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2

    by_id: Dict[str, Any] = {p["arxiv_id"]: p for p in data}

    assert "1111.11111" in by_id
    assert "2222.22222" in by_id

    first = by_id["1111.11111"]
    assert first["title"] == "Test Paper One"
    assert first["abstract"] == "Abstract one"

    second = by_id["2222.22222"]
    assert second["title"] == "Test Paper Two"
    assert second["abstract"] == "Abstract two"


def test_get_paper_endpoint_uses_influence_view_and_supports_404(monkeypatch):
    """
    Ensure GET /papers/{arxiv_id}:

    - Looks up the paper in the in-memory graph
    - Calls get_paper_influence_view with the correct arguments
    - Returns 404 if the paper isn't present
    """
    client = TestClient(app)

    # Start from the same mock graph and add a known paper node
    G = make_mock_graph()
    target_id = "9999.99999"

    G.add_node(
        f"paper:{target_id}",
        arxiv_id=target_id,
        title="Target Paper",
        abstract="Target abstract",
    )

    app.state.graph = G

    captured: Dict[str, Any] = {}

    def fake_get_paper_influence_view(
        G_arg,
        arxiv_id: str,
        top_k_concepts: int,
        top_k_references: int,
        top_k_influenced: int,
    ):
        captured["args"] = {
            "G": G_arg,
            "arxiv_id": arxiv_id,
            "top_k_concepts": top_k_concepts,
            "top_k_references": top_k_references,
            "top_k_influenced": top_k_influenced,
        }
        # Reuse the same dummy influence view shape as ingest tests
        return _make_dummy_influence_view(arxiv_id)

    # Patch the symbol used by the web app for influence view
    monkeypatch.setattr(
        web_app_module,
        "get_paper_influence_view",
        fake_get_paper_influence_view,
        raising=True,
    )

    # ---- Existing paper: should call influence view and return 200 ----
    response = client.get(f"/papers/{target_id}")
    assert response.status_code == 200

    body = response.json()
    assert "paper" in body
    assert body["paper"]["arxiv_id"] == target_id

    # Ensure our fake was called with the right graph and id
    assert captured["args"]["G"] is app.state.graph
    assert captured["args"]["arxiv_id"] == target_id
    # Sanity-check that k-values are present (exact numbers are less important)
    assert isinstance(captured["args"]["top_k_concepts"], int)
    assert isinstance(captured["args"]["top_k_references"], int)
    assert isinstance(captured["args"]["top_k_influenced"], int)

    # ---- Missing paper: should return 404 ----
    missing_id = "0000.00000"
    response_404 = client.get(f"/papers/{missing_id}")
    assert response_404.status_code == 404
    detail = response_404.json().get("detail", "")
    assert "not found" in detail.lower()
