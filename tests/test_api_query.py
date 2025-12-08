# tests/test_api_query.py

import networkx as nx

from kg_ai_papers.api.query import (
    get_paper_concepts,
    get_influential_references,
    get_influenced_papers,
    get_paper_influence_view,
)
from kg_ai_papers.graph.schema import NodeType, EdgeType


def build_toy_graph():
    G = nx.MultiDiGraph()

    # Nodes
    G.add_node(
        "paper:p1",
        type=NodeType.PAPER.value,
        arxiv_id="p1",
        title="Paper One",
        abstract="Abstract one",
    )
    G.add_node(
        "paper:p2",
        type=NodeType.PAPER.value,
        arxiv_id="p2",
        title="Paper Two",
        abstract="Abstract two",
    )
    G.add_node(
        "concept:c1",
        type=NodeType.CONCEPT.value,
        label="graph neural networks",
    )

    # PAPER_HAS_CONCEPT edge
    G.add_edge(
        "paper:p1",
        "concept:c1",
        type=EdgeType.PAPER_HAS_CONCEPT.value,
        weight=0.9,
    )

    # Citation edge p1 -> p2
    G.add_edge(
        "paper:p1",
        "paper:p2",
        type=EdgeType.PAPER_CITES_PAPER.value,
        influence_score=0.8,
        similarity=0.5,
    )

    return G


def test_get_paper_concepts():
    G = build_toy_graph()
    concepts = get_paper_concepts(G, "p1")
    assert len(concepts) == 1
    assert concepts[0].label == "graph neural networks"
    assert concepts[0].weight == 0.9


def test_get_influential_references():
    G = build_toy_graph()
    refs = get_influential_references(G, "p1")
    assert len(refs) == 1
    assert refs[0].arxiv_id == "p2"
    assert refs[0].influence_score == 0.8


def test_get_influenced_papers():
    G = build_toy_graph()
    influenced = get_influenced_papers(G, "p2")
    assert len(influenced) == 1
    assert influenced[0].arxiv_id == "p1"


def test_get_paper_influence_view():
    G = build_toy_graph()
    view = get_paper_influence_view(G, "p1")

    assert view.paper.arxiv_id == "p1"
    assert len(view.concepts) == 1
    assert len(view.influential_references) == 1
    # p1 is not cited by anyone else in toy graph
    assert len(view.influenced_papers) == 0
