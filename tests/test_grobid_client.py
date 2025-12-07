# tests/test_grobid_client.py

from pathlib import Path

import pytest

from kg_ai_papers.grobid_client import GrobidClient, GrobidClientError


@pytest.mark.integration
def test_grobid_healthcheck():
    client = GrobidClient()
    assert client.healthcheck() is True


@pytest.mark.integration
def test_process_pdf_roundtrip(tmp_path: Path):
    # point to a small sample PDF in your repo
    sample_pdf = Path("tests/data/sample_paper.pdf")
    if not sample_pdf.exists():
        pytest.skip("sample_paper.pdf not available")

    client = GrobidClient(tei_dir=tmp_path / "tei")
    tei_path = client.process_pdf(sample_pdf, paper_id="sample_paper")

    assert tei_path.exists()
    text = tei_path.read_text(encoding="utf-8")
    assert "<TEI" in text
    assert "</TEI>" in text
