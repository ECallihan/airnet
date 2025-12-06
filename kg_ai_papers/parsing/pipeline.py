# kg_ai_papers/parsing/pipeline.py

from __future__ import annotations

from kg_ai_papers.models.paper import Paper
from kg_ai_papers.parsing.grobid_client import process_fulltext, GrobidClientError
from kg_ai_papers.parsing.tei_parser import parse_sections, parse_references


def parse_pdf_to_paper(paper: Paper, pdf_path: str) -> None:
    """
    Given a Paper with metadata and a PDF path, enrich it with sections & references.
    """
    try:
        tei_xml = process_fulltext(pdf_path)
    except GrobidClientError as e:
        # Soft failure: log and leave sections/references empty
        print(f"[WARN] GROBID failed for {paper.arxiv_id}: {e}")
        paper.sections = []
        paper.references = []
        return

    sections = parse_sections(tei_xml)
    references = parse_references(tei_xml)

    paper.sections = sections
    paper.references = references
