from dataclasses import dataclass

from kg_ai_papers.tei_parser import PaperSection
from kg_ai_papers.nlp.concept_extraction import (
    SectionConcept,
    ConceptSummary,
    aggregate_section_concepts,
)


@dataclass
class DummyConcept:
    name: str
    kind: str = "method"


def test_aggregate_section_concepts_basic():
    # Two sections with different titles
    intro = PaperSection(
        id="s1",
        title="Introduction",
        level=1,
        text="",
        path=["Introduction"],
    )
    methods = PaperSection(
        id="s2",
        title="Methods",
        level=1,
        text="",
        path=["Methods"],
    )

    c1 = DummyConcept(name="Graph Neural Networks")
    c2 = DummyConcept(name="Graph Neural Networks")  # same concept, different mention

    sc_list = [
        SectionConcept(concept=c1, paper_id="p1", section_title=intro.title, section_level=intro.level),
        SectionConcept(concept=c2, paper_id="p1", section_title=methods.title, section_level=methods.level),
    ]

    summaries = aggregate_section_concepts(sc_list)
    assert "Graph Neural Networks" in summaries

    summary: ConceptSummary = summaries["Graph Neural Networks"]
    assert summary.mentions_total == 2
    assert summary.mentions_by_section["Introduction"] == 1
    assert summary.mentions_by_section["Methods"] == 1

    # With default weights, Methods should have higher weight than Introduction
    assert summary.weighted_score > 1.0
