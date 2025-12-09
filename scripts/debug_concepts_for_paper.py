# scripts/debug_concepts_for_paper.py

from kg_ai_papers.graph.storage import load_latest_graph
from kg_ai_papers.api import graph_api as ga

import sys

def main(arxiv_id: str) -> None:
    G = load_latest_graph()
    print(f"[debug] Loaded graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node = ga._find_paper_node(G, arxiv_id)
    if node is None:
        print(f"[debug] No node found for arxiv_id={arxiv_id}")
        return

    print(f"[debug] Paper node id = {node}, attrs = {G.nodes[node]}")

    concepts = ga._concepts_for_paper(G, node)
    print(f"[debug] Found {len(concepts)} concept neighbors")

    for c in concepts[:20]:
        print(
            f"  - concept id={c.id!r}, label={c.label!r}, "
            f"importance={c.importance}, mentions_total={c.mentions_total}"
        )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.debug_concepts_for_paper <arxiv_id>")
        raise SystemExit(1)
    main(sys.argv[1])
