# tests/test_influence.py

import networkx as nx

from kg_ai_papers.nlp.concept_extraction import ConceptSummary
from kg_ai_papers.graph.influence import (
    compute_influence_features,
    attach_influence_scores_to_graph,
)


def test_compute_influence_features_basic():
    paper_concepts = {
        "A": {
            "GNN": ConceptSummary(
                concept_key="GNN",
                base_name="Graph Neural Networks",
                kind="method",
                mentions_total=3,
                mentions_by_section={"Methods": 3},
                weighted_score=3.0,
            ),
            "LoRA": ConceptSummary(
                concept_key="LoRA",
                base_name="LoRA",
                kind="method",
                mentions_total=1,
                mentions_by_section={"Introduction": 1},
                weighted_score=1.0,
            ),
        },
        "B": {
            "GNN": ConceptSummary(
                concept_key="GNN",
                base_name="Graph Neural Networks",
                kind="method",
                mentions_total=2,
                mentions_by_section={"Methods": 2},
                weighted_score=2.0,
            ),
        },
    }

    feats = compute_influence_features("A", "B", paper_concept_summaries=paper_concepts)
    assert feats is not None
    assert feats.num_shared_concepts == 1
    assert feats.jaccard_unweighted > 0
    assert feats.jaccard_weighted > 0
    assert 0 <= feats.influence_score <= 1


def test_attach_influence_scores_to_graph():
    g = nx.MultiDiGraph()
    g.add_node("paper:A", paper_id="A", type="paper")
    g.add_node("paper:B", paper_id="B", type="paper")

    g.add_edge("paper:A", "paper:B", key=0, type="CITES")

    # Re-use the same concept summaries as above
    paper_concepts = {
        "A": {
            "GNN": ConceptSummary(
                concept_key="GNN",
                base_name="Graph Neural Networks",
                kind="method",
                mentions_total=3,
                mentions_by_section={"Methods": 3},
                weighted_score=3.0,
            )
        },
        "B": {
            "GNN": ConceptSummary(
                concept_key="GNN",
                base_name="Graph Neural Networks",
                kind="method",
                mentions_total=2,
                mentions_by_section={"Methods": 2},
                weighted_score=2.0,
            )
        },
    }

    attach_influence_scores_to_graph(g, paper_concepts)

    data = next(iter(g.get_edge_data("paper:A", "paper:B").values()))
    assert "influence_score" in data
    assert data["influence_score"] > 0
