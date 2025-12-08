# tests/test_graph_builder.py

from kg_ai_papers.models.paper import Paper
from kg_ai_papers.graph.builder import build_graph
from kg_ai_papers.graph.schema import NodeType, EdgeType


class DummyEmb:
    """Tiny dummy object acting like a vector; we won't actually compute similarity here."""
    def __init__(self, v):
        self.v = v


def test_build_graph_basic(monkeypatch):
    # Prepare two simple papers
    p1 = Paper(
        arxiv_id="p1",
        title="Paper One",
        abstract="Graph neural networks.",
    )
    p2 = Paper(
        arxiv_id="p2",
        title="Paper Two",
        abstract="Reinforcement learning.",
    )

    # Fake paper-level concepts
    p1.paper_level_concepts = [("graph neural networks", 0.9)]
    p2.paper_level_concepts = [("reinforcement learning", 0.8)]

    # Fake embeddings as simple small vectors
    import torch
    p1.embedding = torch.tensor([1.0, 0.0])
    p2.embedding = torch.tensor([0.0, 1.0])

    papers = [p1, p2]
    citation_map = {
        "p1": ["p2"],  # p1 cites p2
        "p2": [],
    }

    G = build_graph(papers, citation_map)

    # Check paper nodes
    paper_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == NodeType.PAPER.value
    ]
    assert len(paper_nodes) == 2

    # Check concept nodes
    concept_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == NodeType.CONCEPT.value
    ]
    assert len(concept_nodes) == 2

    # Check PAPER_HAS_CONCEPT edges
    has_concept_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("type") == EdgeType.PAPER_HAS_CONCEPT.value
    ]
    assert len(has_concept_edges) == 2

    # Check citation edge
    cite_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("type") == EdgeType.PAPER_CITES_PAPER.value
    ]
    assert len(cite_edges) == 1
    (u, v, d) = cite_edges[0]
    assert "influence_score" in d
    assert "similarity" in d
