# tests/test_ingest_pipeline.py

from pathlib import Path
from typing import Any, List

import pytest

from kg_ai_papers.ingest.pipeline import (
    IngestedPaper,
    ingest_pdf,
    persist_ingested_paper_to_neo4j,
)
from kg_ai_papers.nlp.concept_extraction import SectionConcept, ConceptSummary


class DummyGrobidClient:
    def __init__(self, tei_path: Path):
        self.tei_path = tei_path
        self.calls: List[Path] = []

    def process_pdf(self, pdf_path: Path) -> Path:
        self.calls.append(pdf_path)
        return self.tei_path


class DummySection:
    def __init__(self, title: str, text: str, level: int = 1):
        self.title = title
        self.text = text
        self.level = level


class FakeNeo4jSession:
    def __init__(self) -> None:
        self.calls = []  # list of (cypher, params)

    def run(self, cypher: str, **params: Any) -> None:
        self.calls.append((cypher, params))


def test_ingest_pdf_wires_components(monkeypatch, tmp_path):
    """
    Ensure ingest_pdf:
      - calls GrobidClient.process_pdf
      - calls extract_sections_from_tei with the TEI path
      - calls extract_concepts_from_sections with the sections + paper_id
      - calls aggregate_section_concepts
      - returns an IngestedPaper with the expected content
    """
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    tei_path = tmp_path / "paper.tei.xml"
    client = DummyGrobidClient(tei_path=tei_path)

    # --- monkeypatch the functions the pipeline imports ---

    # 1) extract_sections_from_tei
    def fake_extract_sections(p):
        assert p == tei_path
        return [DummySection("Introduction", "graph neural networks")]

    monkeypatch.setattr(
        "kg_ai_papers.ingest.pipeline.extract_sections_from_tei",
        fake_extract_sections,
    )

    # 2) extract_concepts_from_sections
    def fake_extract_concepts(sections, paper_id=None):
        # basic contract check
        assert len(sections) == 1
        assert paper_id == "p1"
        return [
            SectionConcept(
                concept="Graph Neural Networks",
                paper_id=paper_id,
                section_title="Introduction",
                section_level=1,
            )
        ]

    monkeypatch.setattr(
        "kg_ai_papers.ingest.pipeline.extract_concepts_from_sections",
        fake_extract_concepts,
    )

    # 3) aggregate_section_concepts
    def fake_aggregate(section_concepts):
        assert len(section_concepts) == 1
        sc = section_concepts[0]
        assert sc.concept == "Graph Neural Networks"
        return {
            "Graph Neural Networks": ConceptSummary(
                concept_key="Graph Neural Networks",
                base_name="Graph Neural Networks",
                kind="method",
                mentions_total=1,
                mentions_by_section={"Introduction": 1},
                weighted_score=0.9,
            )
        }

    monkeypatch.setattr(
        "kg_ai_papers.ingest.pipeline.aggregate_section_concepts",
        fake_aggregate,
    )

    # --- run the pipeline ---
    ingested = ingest_pdf(
        pdf_path=pdf_path,
        paper_id="p1",
        grobid_client=client,
    )

    assert isinstance(ingested, IngestedPaper)
    assert ingested.paper_id == "p1"
    assert ingested.pdf_path == pdf_path
    assert ingested.tei_path == tei_path

    # 1 section, 1 concept
    assert len(ingested.sections) == 1
    assert len(ingested.section_concepts) == 1
    assert "Graph Neural Networks" in ingested.concept_summaries

    # GrobidClient was actually called
    assert client.calls == [pdf_path]


def test_persist_ingested_paper_to_neo4j(monkeypatch, tmp_path):
    """
    Ensure Neo4j persistence:
      - writes the Paper node
      - writes Concept nodes + MENTIONS edges with the right params
      (We don't assert exact Cypher text, only that the parameters look right.)
    """
    pdf_path = tmp_path / "paper.pdf"
    tei_path = tmp_path / "paper.tei.xml"

    summaries = {
        "Graph Neural Networks": ConceptSummary(
            concept_key="Graph Neural Networks",
            base_name="Graph Neural Networks",
            kind="method",
            mentions_total=3,
            mentions_by_section={"Introduction": 1, "Methods": 2},
            weighted_score=4.5,
        )
    }

    ingested = IngestedPaper(
        paper_id="p1",
        pdf_path=pdf_path,
        tei_path=tei_path,
        sections=[DummySection("Introduction", "graph neural networks")],
        section_concepts=[
            SectionConcept(
                concept="Graph Neural Networks",
                paper_id="p1",
                section_title="Introduction",
                section_level=1,
            )
        ],
        concept_summaries=summaries,
    )

    session = FakeNeo4jSession()

    persist_ingested_paper_to_neo4j(session, ingested)

    # We expect two .run() calls: one for the Paper, one for concepts + edges
    assert len(session.calls) == 2

    # First call: Paper node
    cypher1, params1 = session.calls[0]
    assert params1["arxiv_id"] == "p1"
    assert params1["pdf_path"] == str(pdf_path)
    assert params1["tei_path"] == str(tei_path)

    # Second call: concepts + MENTIONS
    cypher2, params2 = session.calls[1]
    assert params2["arxiv_id"] == "p1"
    concepts = params2["concepts"]
    assert isinstance(concepts, list)
    assert len(concepts) == 1
    c0 = concepts[0]
    assert c0["concept_key"] == "Graph Neural Networks"
    assert c0["name"] == "Graph Neural Networks"
    assert c0["kind"] == "method"
    assert c0["mentions_total"] == 3
    assert c0["weighted_score"] == 4.5
    assert c0["mentions_by_section"] == {"Introduction": 1, "Methods": 2}
