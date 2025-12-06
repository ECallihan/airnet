# kg_ai_papers/parsing/pipeline.py

from kg_ai_papers.models.paper import Paper
from kg_ai_papers.parsing.grobid_client import process_fulltext
from kg_ai_papers.parsing.tei_parser import parse_sections, parse_references


def parse_pdf_to_paper(paper: Paper, pdf_path: str) -> Paper:
    """
    Given a Paper with metadata and a PDF path, enrich it with sections & references.
    """
    tei_xml = process_fulltext(pdf_path)
    sections = parse_sections(tei_xml)
    references = parse_references(tei_xml)

    paper.sections = sections
    paper.references = references
    return paper
