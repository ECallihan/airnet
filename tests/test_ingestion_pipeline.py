# tests/test_ingestion_pipeline.py

from pathlib import Path

import networkx as nx
import pytest

from kg_ai_papers.ingest.pipeline import IngestedPaper, ingest_pdf_and_update_graph
from kg_ai_papers.nlp.concept_extraction import ConceptSummary
from kg_ai_papers.models.paper import Paper


def test_ingest_pdf_and_update_graph_integration(monkeypatch, tmp_path):
    """
    Integration-style unit test for the ingestion helper.

    We stub out `ingest_pdf` so we don't need a real GROBID instance or
    real PDF/TEI, and verify that:

    - ingest_pdf_and_update_graph returns an IngestedPaperResult
    - the paper node is added to the graph
    - a concept from `concept_summaries` is added as a concept node
      and wired to the paper.
    """

    # --- Arrange ---------------------------------------------------------

    # Fake PDF/TEI paths inside a pytest tmp_path
    pdf_path = tmp_path / "dummy.pdf"
    tei_path = tmp_path / "dummy.tei.xml"
    pdf_path.write_bytes(b"%PDF-FAKE")      # just to exist on disk
    tei_path.write_text("<TEI>fake</TEI>")  # not actually parsed in this test

    arxiv_id = "9999.00001"
    concept_key = "test concept"

    # Minimal concept summary; fields without defaults must be provided
    fake_summary = ConceptSummary(
        concept_key=concept_key,
        base_name=concept_key,
        kind=None,
        mentions_total=1,
        weighted_score=0.9,
    )

    # IngestedPaper stub with just enough fields for the graph builder
    fake_ingested = IngestedPaper(
        arxiv_id=arxiv_id,
        pdf_path=pdf_path,
        tei_path=tei_path,
        sections=[],          # sections aren't used by update_graph_with_ingested_paper
        section_concepts=[],  # can be empty for this test
        concept_summaries={concept_key: fake_summary},
    )

    # Monkeypatch ingest_pdf so ingestion is "instant" and offline
    def fake_ingest_pdf(*args, **kwargs):
        return fake_ingested

    import kg_ai_papers.ingest.pipeline as ingest_pipeline

    monkeypatch.setattr(ingest_pipeline, "ingest_pdf", fake_ingest_pdf)

    # Prepare an empty graph and a Paper instance (like the CLI does now)
    G = nx.MultiDiGraph()
    paper = Paper(
        arxiv_id=arxiv_id,
        title="Dummy title for testing",
        abstract="Dummy abstract for testing",
    )

    # --- Act -------------------------------------------------------------

    result = ingest_pdf_and_update_graph(
        G=G,
        pdf_path=pdf_path,
        grobid_client=None,   # not used by our fake_ingest_pdf
        paper=paper,
        neo4j_session=None,
    )

    # --- Assert: function return -----------------------------------------

    from kg_ai_papers.ingest.pipeline import IngestedPaperResult

    assert isinstance(result, IngestedPaperResult)
    assert result.paper.arxiv_id == arxiv_id
    assert concept_key in result.concept_summaries

    # --- Assert: graph structure -----------------------------------------

    # The paper node id is just the arxiv_id, per update_graph_with_ingested_paper
    paper_node_id = arxiv_id
    concept_node_id = f"concept::{concept_key}"

    # Paper node exists
    assert paper_node_id in G.nodes
    assert G.nodes[paper_node_id]["type"] == "paper"
    assert G.nodes[paper_node_id]["arxiv_id"] == arxiv_id

    # Concept node exists
    assert concept_node_id in G.nodes
    assert G.nodes[concept_node_id]["type"] == "concept"
    assert G.nodes[concept_node_id]["key"] == concept_key

    # Edge between paper and concept exists with correct type
    edges = list(G.edges(paper_node_id, data=True))
    assert any(
        tgt == concept_node_id and data.get("type") == "MENTIONS"
        for (_, tgt, data) in edges
    )
