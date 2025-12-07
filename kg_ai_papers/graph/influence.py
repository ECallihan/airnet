# kg_ai_papers/graph/influence.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import networkx as nx

from kg_ai_papers.nlp.concept_extraction import ConceptSummary


ConceptSummaryMap = Mapping[str, ConceptSummary]  # concept_key -> ConceptSummary
PaperConceptSummaries = Mapping[str, ConceptSummaryMap]  # paper_id -> concept_key -> ConceptSummary


@dataclass
class InfluenceFeatures:
    """
    Features describing how much one paper may be influenced by another.
    All scores are in [0, 1] where possible.
    """
    citing_paper_id: str
    cited_paper_id: str

    num_shared_concepts: int
    citing_concept_count: int
    cited_concept_count: int

    jaccard_unweighted: float
    jaccard_weighted: float

    # Final scalar we’ll attach to the graph edge
    influence_score: float


def _compute_concept_overlap(
    citing: ConceptSummaryMap,
    cited: ConceptSummaryMap,
) -> Tuple[int, int, int, float, float]:
    """
    Compute concept overlap stats between two papers.

    Returns:
        (num_shared, citing_total, cited_total, jaccard_unweighted, jaccard_weighted)
    """
    citing_keys = set(citing.keys())
    cited_keys = set(cited.keys())

    shared = citing_keys & cited_keys
    num_shared = len(shared)

    citing_total = len(citing_keys)
    cited_total = len(cited_keys)

    # Unweighted Jaccard on concept sets
    union_size = len(citing_keys | cited_keys)
    jaccard_unweighted = (num_shared / union_size) if union_size > 0 else 0.0

    # Weighted Jaccard using ConceptSummary.weighted_score
    # Treat each paper as a vector over concept_key -> weighted_score
    if not shared:
        jaccard_weighted = 0.0
    else:
        shared_min_sum = 0.0
        union_max_sum = 0.0

        all_keys = citing_keys | cited_keys
        for k in all_keys:
            w_citing = citing.get(k).weighted_score if k in citing else 0.0
            w_cited = cited.get(k).weighted_score if k in cited else 0.0

            union_max_sum += max(w_citing, w_cited)
            if k in shared:
                shared_min_sum += min(w_citing, w_cited)

        jaccard_weighted = (shared_min_sum / union_max_sum) if union_max_sum > 0 else 0.0

    return num_shared, citing_total, cited_total, jaccard_unweighted, jaccard_weighted


def compute_influence_features(
    citing_paper_id: str,
    cited_paper_id: str,
    paper_concept_summaries: PaperConceptSummaries,
) -> Optional[InfluenceFeatures]:
    """
    Compute InfluenceFeatures for a (citing -> cited) pair using ConceptSummary maps.
    Returns None if either paper has no concept summaries.
    """
    citing = paper_concept_summaries.get(citing_paper_id)
    cited = paper_concept_summaries.get(cited_paper_id)

    if not citing or not cited:
        return None

    (
        num_shared,
        citing_total,
        cited_total,
        jaccard_unweighted,
        jaccard_weighted,
    ) = _compute_concept_overlap(citing, cited)

    # Base influence score: mostly driven by weighted Jaccard,
    # with a small bonus for unweighted overlap.
    influence_score = (0.8 * jaccard_weighted) + (0.2 * jaccard_unweighted)

    return InfluenceFeatures(
        citing_paper_id=citing_paper_id,
        cited_paper_id=cited_paper_id,
        num_shared_concepts=num_shared,
        citing_concept_count=citing_total,
        cited_concept_count=cited_total,
        jaccard_unweighted=jaccard_unweighted,
        jaccard_weighted=jaccard_weighted,
        influence_score=influence_score,
    )


def attach_influence_scores_to_graph(
    graph: nx.MultiDiGraph,
    paper_concept_summaries: PaperConceptSummaries,
    *,
    citation_edge_type: str = "CITES",
    influence_attr: str = "influence_score",
) -> nx.MultiDiGraph:
    """
    Iterate over citation edges in the graph and annotate them with influence scores.

    Expects edges of the form:
        graph.add_edge(u, v, type="CITES", paper_id_u=..., paper_id_v=...)

    or at least that node attributes contain 'paper_id' for u and v.

    Args:
        graph: The graph to mutate.
        paper_concept_summaries: Dict[paper_id] -> Dict[concept_key] -> ConceptSummary.
        citation_edge_type: Edge 'type' attribute to treat as citations.
        influence_attr: Attribute name to store the final scalar score under.

    Returns:
        The same graph instance, for convenience.
    """
    # We’ll walk all edges and find citation edges
    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get("type") != citation_edge_type:
            continue

        # Try to infer paper ids from node attributes; fall back to node ids themselves
        citing_paper_id = graph.nodes[u].get("paper_id", u)
        cited_paper_id = graph.nodes[v].get("paper_id", v)

        feats = compute_influence_features(
            citing_paper_id=citing_paper_id,
            cited_paper_id=cited_paper_id,
            paper_concept_summaries=paper_concept_summaries,
        )
        if feats is None:
            # No concept summaries; skip
            continue

        # Attach full feature set onto the edge
        data[influence_attr] = feats.influence_score
        data["influence_num_shared_concepts"] = feats.num_shared_concepts
        data["influence_jaccard_unweighted"] = feats.jaccard_unweighted
        data["influence_jaccard_weighted"] = feats.jaccard_weighted

    return graph
