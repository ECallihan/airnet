import networkx as nx
from fastapi.testclient import TestClient

from kg_ai_papers.api.graph_api import app
from kg_ai_papers.ingest.pipeline import IngestedPaperResult
from kg_ai_papers.models.paper import Paper


def _make_ingested_result(arxiv_id: str) -> IngestedPaperResult:
    paper = Paper(
        arxiv_id=arxiv_id,
        title=f"Title for {arxiv_id}",
        abstract=f"Abstract for {arxiv_id}",
        pdf_path=f"/tmp/{arxiv_id}.pdf",
    )

    class DummyConcept:
        def __init__(self, score: float) -> None:
            self.score = score
            self.section_weight = 1.0

    concept_summaries = {
        "concept:one": DummyConcept(0.9),
        "concept:two": DummyConcept(0.5),
    }
    references = ["ref-1", "ref-2"]

    return IngestedPaperResult(
        paper=paper,
        concept_summaries=concept_summaries,
        references=references,
    )


def test_ingest_arxiv_batch_success(monkeypatch, tmp_path):
    import kg_ai_papers.api.graph_api as graph_api_mod

    G = nx.MultiDiGraph()

    def fake_get_graph():
        return G

    monkeypatch.setattr(graph_api_mod, "get_graph", fake_get_graph)

    # Avoid real file writes
    saved_paths = []

    def fake_save_graph(graph, path):
        saved_paths.append(str(path))

    monkeypatch.setattr(graph_api_mod, "save_graph", fake_save_graph)

    # Minimal update_graph stub
    update_calls = []

    def fake_update_graph(graph, result):
        update_calls.append(result.paper.arxiv_id)
        node_id = f"paper:{result.paper.arxiv_id}"
        graph.add_node(node_id, kind="paper", arxiv_id=result.paper.arxiv_id)

    monkeypatch.setattr(
        graph_api_mod,
        "update_graph_with_ingested_paper",
        fake_update_graph,
    )

    # Fake ingestion pipeline
    def fake_ingest_arxiv_paper(*, arxiv_id, work_dir, use_cache=True, force_reingest=False):
        return _make_ingested_result(arxiv_id)

    monkeypatch.setattr(graph_api_mod, "ingest_arxiv_paper", fake_ingest_arxiv_paper)

    client = TestClient(app)

    payload = {
        "ids": ["1234.0001", "1234.0002", "1234.0001"],  # includes a duplicate
        "work_dir": str(tmp_path / "ingest"),
        "use_cache": True,
        "force_reingest": False,
    }

    resp = client.post("/ingest/arxiv", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Duplicate should be removed
    assert data["requested_ids"] == ["1234.0001", "1234.0002"]
    assert len(data["ingested"]) == 2
    assert data["failed"] == []

    ids = {p["arxiv_id"] for p in data["ingested"]}
    assert ids == {"1234.0001", "1234.0002"}

    for p in data["ingested"]:
        assert p["num_concepts"] == 2
        assert p["num_references"] == 2

    # We added 2 paper nodes
    assert data["graph_num_nodes"] == 2
    assert data["graph_num_edges"] == 0

    assert update_calls == ["1234.0001", "1234.0002"]
    assert len(saved_paths) == 1


def test_ingest_arxiv_batch_all_fail(monkeypatch, tmp_path):
    import kg_ai_papers.api.graph_api as graph_api_mod

    G = nx.MultiDiGraph()

    def fake_get_graph():
        return G

    monkeypatch.setattr(graph_api_mod, "get_graph", fake_get_graph)

    def fake_save_graph(graph, path):
        raise AssertionError("save_graph should not be called when all ingests fail")

    monkeypatch.setattr(graph_api_mod, "save_graph", fake_save_graph)

    # Ingestion always fails
    def fake_ingest_arxiv_paper(*, arxiv_id, work_dir, use_cache=True, force_reingest=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(graph_api_mod, "ingest_arxiv_paper", fake_ingest_arxiv_paper)

    client = TestClient(app)

    payload = {
        "ids": ["bad-1", "bad-2"],
        "work_dir": str(tmp_path / "ingest"),
        "use_cache": True,
        "force_reingest": False,
    }

    resp = client.post("/ingest/arxiv", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"] == "No papers were successfully ingested."
