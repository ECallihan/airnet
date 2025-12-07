# tests/test_concept_extraction.py
from typing import Iterable, List, Optional

from kg_ai_papers.config.settings import Settings, RuntimeMode
from kg_ai_papers.models.paper import Paper
from kg_ai_papers.models.section import Section
from kg_ai_papers.nlp.concept_extraction import ConceptExtractor

_settings = Settings()

# Section kinds/titles to prioritize in LIGHT mode.
# This is just a heuristic – you can refine it later.
_LIGHT_MODE_IMPORTANT_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "conclusion",
    "summary",
}


def test_concept_extraction_basic():
    paper = Paper(
        arxiv_id="test-0001",
        title="Graph Neural Networks for AI",
        abstract="We propose a simple graph neural network for AI tasks.",
    )
    paper.sections = [
        Section(
            title="Introduction",
            level="1",
            text="Graph neural networks have become a powerful tool for AI. "
                 "They are used for node classification and link prediction.",
        ),
        Section(
            title="Method",
            level="1",
            text="Our graph neural network uses message passing and attention.",
        ),
    ]

    extractor = ConceptExtractor(model_name=None)  # KeyBERT default
    extractor.extract_for_paper(paper, section_top_n=5, paper_top_n=10)

    # Section-level occurrences should exist
    assert len(paper.concepts) > 0

    # Paper-level concepts should exist and be sorted
    assert len(paper.paper_level_concepts) > 0
    # Sorted by descending score
    scores = [score for _, score in paper.paper_level_concepts]
    assert scores == sorted(scores, reverse=True)
