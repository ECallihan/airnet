# tests/test_graph_builder_concepts.py

import networkx as nx

from kg_ai_papers.nlp.concept_extraction import ConceptSummary
from kg_ai_papers.graph.builder import (
    attach_concepts_to_graph,
    paper_node_id,
    concept_node_id,
)


def test_attach_concepts_to_graph_basic():
    g = nx.MultiDiGraph()
    paper_id = "paper-1"

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

    attach_concepts_to_graph(g, {paper_id: summaries})

    p_node = paper_node_id(paper_id)
    c_node = concept_node_id("Graph Neural Networks")

    assert p_node in g
    assert c_node in g

    # For MultiDiGraph, get_edge_data returns a dict: {key -> edge_attr_dict}
    edge_dict = g.get_edge_data(p_node, c_node)
    assert edge_dict is not None
    assert len(edge_dict) == 1

    # Grab the first (and only) edge's attributes
    data = next(iter(edge_dict.values()))
    assert data["type"] == "MENTIONS_CONCEPT"
    assert data["mentions_total"] == 3
    assert data["mentions_by_section"]["Methods"] == 2
    assert data["weighted_score"] == 4.5
