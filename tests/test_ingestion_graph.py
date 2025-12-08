# tests/test_ingestion_graph.py

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from kg_ai_papers.graph.builder import (
    update_graph_with_ingested_paper,
    paper_node_id,
    concept_node_id,
    get_paper_concept_edges,
)
from kg_ai_papers.graph.schema import EdgeType, NodeType
from kg_ai_papers.models.paper import Paper
from kg_ai_papers.nlp.concept_extraction import ConceptSummary


@dataclass
class IngestedPaperResult:
    """
    Minimal duck-typed stand-in for the real ingestion result.

    We keep it local to the test so this test only depends on the
    *shape* of the object, not the implementation details in
    kg_ai_papers.ingest.pipeline.
    """
    paper: Paper
    concept_summaries: Dict[str, ConceptSummary]
    references: List[str]


def test_update_graph_with_ingested_paper_basic():
    # 1) Build a fake ingestion result
    paper = Paper(
        arxiv_id="p_ing_1",
        title="Graph Neural Networks in Practice",
        abstract="We explore graph neural networks for AI research papers.",
    )

    concept_summaries = {
        "c1": ConceptSummary(
            concept_key="c1",
            base_name="graph neural networks",
            kind=None,
            mentions_total=3,
            mentions_by_section={"Introduction": 2, "Methods": 1},
            weighted_score=0.9,
        )
    }

    references = ["p_ing_2"]

    result = IngestedPaperResult(
        paper=paper,
        concept_summaries=concept_summaries,
        references=references,
    )

    # 2) Start from an empty graph and integrate the ingestion result
    G = nx.MultiDiGraph()
    update_graph_with_ingested_paper(G, result)

    # 3) Base paper node (unprefixed) should exist and be typed correctly
    assert "p_ing_1" in G
    p_data = G.nodes["p_ing_1"]
    assert p_data["type"] == NodeType.PAPER.value
    assert p_data["title"] == paper.title
    assert p_data["abstract"] == paper.abstract

    # 4) Citation edge from p_ing_1 -> p_ing_2 should exist
    assert "p_ing_2" in G
    edges_p1_p2 = G.get_edge_data("p_ing_1", "p_ing_2") or {}
    assert len(edges_p1_p2) >= 1
    # at least one edge of type PAPER_CITES_PAPER
    assert any(
        d.get("type") == EdgeType.PAPER_CITES_PAPER.value
        for d in edges_p1_p2.values()
    )

    # 5) Concept-layer nodes should also exist using the helper IDs
    p_concept_node = paper_node_id("p_ing_1")
    c_node = concept_node_id("c1")

    assert p_concept_node in G
    assert c_node in G

    # Check that there is at least one MENTIONS_CONCEPT edge with the right stats
    concept_edges = get_paper_concept_edges(G, "p_ing_1", "c1")
    assert len(concept_edges) == 1
    edge_attrs = concept_edges[0]
    assert edge_attrs["mentions_total"] == 3
    assert edge_attrs["weighted_score"] == 0.9
    assert edge_attrs["mentions_by_section"] == {"Introduction": 2, "Methods": 1}
