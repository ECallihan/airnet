# tests/test_web_ingest.py

from pathlib import Path

import networkx as nx
from fastapi.testclient import TestClient

import kg_ai_papers.web.app as web_app_module
from kg_ai_papers.web.app import app
from kg_ai_papers.api import query as api_query  # noqa: F401 (kept for context)
from kg_ai_papers.ingest import pipeline as ingest_pipeline  # noqa: F401
from kg_ai_papers.api.models import PaperInfluenceResult, PaperSummary


class DummyNeo4jSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def make_mock_graph():
    return nx.MultiDiGraph()


def _make_dummy_influence_view(arxiv_id: str) -> PaperInfluenceResult:
    return PaperInfluenceResult(
        paper=PaperSummary(arxiv_id=arxiv_id, title="Dummy title", abstract=None),
        concepts=[],
        influential_references=[],
        influenced_papers=[],
    )


def test_ingest_arxiv_endpoint_wires_neo4j_and_persists_graph(monkeypatch, tmp_path):
    client = TestClient(app)

    # Use a fresh in-memory graph
    G = make_mock_graph()
    app.state.graph = G

    captured = {}

    # 1) Neo4j session helper (module-level in web.app)
    def fake_get_neo4j_session_or_none():
        s = DummyNeo4jSession()
        captured["neo4j_session"] = s
        return s

    monkeypatch.setattr(
        web_app_module,
        "_get_neo4j_session_or_none",
        fake_get_neo4j_session_or_none,
        raising=True,
    )

    # 2) ingest_arxiv_paper - patch the symbol used by the endpoint
    from kg_ai_papers.ingest.pipeline import IngestedPaperResult
    from kg_ai_papers.models.paper import Paper

    def fake_ingest_arxiv_paper(
        *,
        arxiv_id,
        work_dir,
        grobid_client=None,
        neo4j_session=None,
        references=None,
    ):
        captured["ingest_arxiv_kwargs"] = {
            "arxiv_id": arxiv_id,
            "work_dir": work_dir,
            "neo4j_session": neo4j_session,
        }
        # Paper requires arxiv_id, title, abstract
        paper = Paper(
            arxiv_id=arxiv_id,
            title=f"Dummy title for {arxiv_id}",
            abstract=None,
        )
        return IngestedPaperResult(
            paper=paper,
            concept_summaries={},
            references=[],
        )

    monkeypatch.setattr(
        web_app_module,
        "ingest_arxiv_paper",  # this is the imported name in web.app
        fake_ingest_arxiv_paper,
        raising=True,
    )

    # 3) Graph persistence - patch symbol used in web.app
    saved = {}

    def fake_save_graph(G_arg, name=None):
        saved["called_with"] = (G_arg, name)
        return tmp_path / "graph.gpickle"

    monkeypatch.setattr(
        web_app_module,
        "save_graph",
        fake_save_graph,
        raising=True,
    )

    # 4) Influence view - patch symbol used in web.app
    def fake_get_paper_influence_view(
        G_arg,
        arxiv_id,
        top_k_concepts,
        top_k_references,
        top_k_influenced,
    ):
        captured["get_paper_influence_view_args"] = {
            "G": G_arg,
            "arxiv_id": arxiv_id,
        }
        return _make_dummy_influence_view(arxiv_id)

    monkeypatch.setattr(
        web_app_module,
        "get_paper_influence_view",
        fake_get_paper_influence_view,
        raising=True,
    )

    # --- Call the endpoint ---
    arxiv_id = "1234.56789"
    response = client.post("/ingest/arxiv", json={"arxiv_id": arxiv_id})
    assert response.status_code == 200
    data = response.json()

    # Response content
    assert data["paper"]["arxiv_id"] == arxiv_id

    # Neo4j session is created and passed into ingest_arxiv_paper
    assert "neo4j_session" in captured
    assert "ingest_arxiv_kwargs" in captured
    assert captured["ingest_arxiv_kwargs"]["neo4j_session"] is captured["neo4j_session"]

    # Graph is saved with our in-memory graph
    assert "called_with" in saved
    saved_graph, saved_name = saved["called_with"]
    assert saved_graph is G


def test_ingest_pdf_endpoint_wires_neo4j_and_persists_graph(monkeypatch, tmp_path):
    client = TestClient(app)

    G = make_mock_graph()
    app.state.graph = G

    captured = {}

    # 1) Neo4j session helper
    def fake_get_neo4j_session_or_none():
        s = DummyNeo4jSession()
        captured["neo4j_session"] = s
        return s

    monkeypatch.setattr(
        web_app_module,
        "_get_neo4j_session_or_none",
        fake_get_neo4j_session_or_none,
        raising=True,
    )

    # 2) ingest_pdf_and_update_graph - patch symbol used in web.app
    from kg_ai_papers.ingest.pipeline import IngestedPaperResult
    from kg_ai_papers.models.paper import Paper

    def fake_ingest_pdf_and_update_graph(
        G_arg,
        pdf_path,
        *,
        grobid_client=None,
        neo4j_session=None,
        paper=None,
        arxiv_id=None,
        references=None,
    ):
        captured["ingest_pdf_kwargs"] = {
            "pdf_path": Path(pdf_path),
            "neo4j_session": neo4j_session,
        }
        # default id from filename stem if not given
        pid = arxiv_id or Path(pdf_path).stem
        p = Paper(
            arxiv_id=pid,
            title=f"Dummy title for {pid}",
            abstract=None,
        )
        # update graph
        G_arg.add_node(f"paper:{pid}", arxiv_id=pid)
        return IngestedPaperResult(
            paper=p,
            concept_summaries={},
            references=[],
        )

    monkeypatch.setattr(
        web_app_module,
        "ingest_pdf_and_update_graph",
        fake_ingest_pdf_and_update_graph,
        raising=True,
    )

    # 3) Graph persistence
    saved = {}

    def fake_save_graph(G_arg, name=None):
        saved["called_with"] = (G_arg, name)
        return tmp_path / "graph.gpickle"

    monkeypatch.setattr(
        web_app_module,
        "save_graph",
        fake_save_graph,
        raising=True,
    )

    # 4) Influence view
    def fake_get_paper_influence_view(
        G_arg,
        arxiv_id,
        top_k_concepts,
        top_k_references,
        top_k_influenced,
    ):
        captured["get_paper_influence_view_args"] = {
            "G": G_arg,
            "arxiv_id": arxiv_id,
        }
        return _make_dummy_influence_view(arxiv_id)

    monkeypatch.setattr(
        web_app_module,
        "get_paper_influence_view",
        fake_get_paper_influence_view,
        raising=True,
    )

    # --- Call the endpoint with a fake PDF upload ---
    files = {"file": ("sample.pdf", b"%PDF-1.4 fake content", "application/pdf")}
    response = client.post("/ingest/pdf", files=files)
    assert response.status_code == 200
    data = response.json()

    # The logical paper id is the filename stem by default
    assert data["paper"]["arxiv_id"] == "sample"

    # Neo4j session passed into ingest_pdf_and_update_graph
    assert "ingest_pdf_kwargs" in captured
    assert captured["ingest_pdf_kwargs"]["neo4j_session"] is captured["neo4j_session"]

    # Graph is saved with our in-memory graph
    assert "called_with" in saved
    saved_graph, saved_name = saved["called_with"]
    assert saved_graph is G
