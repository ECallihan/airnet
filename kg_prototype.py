#!/usr/bin/env python3
"""
kg_prototype.py

Minimal prototype for:
- Fetching a small set of papers from arXiv
- Computing sentence embeddings
- Building a small similarity-based knowledge graph
- Running a few example "explain this paper" queries

Dependencies (install with pip if needed):
    pip install arxiv sentence-transformers networkx numpy
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import arxiv
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Fetching from arXiv (using the modern Client API)
# ---------------------------------------------------------------------------

def fetch_papers(
    query: str,
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
) -> List[Paper]:
    """
    Fetch a small set of papers from arXiv using the non-deprecated Client.results API.
    """
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3,
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
    )

    papers: List[Paper] = []

    for result in client.results(search):
        # get_short_id() gives things like "2101.00001"
        if hasattr(result, "get_short_id"):
            arxiv_id = result.get_short_id()
        else:
            # Fallback: parse from entry_id URL
            arxiv_id = result.entry_id.rsplit("/", 1)[-1]

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=result.title.strip(),
                abstract=result.summary.strip(),
                pdf_url=result.pdf_url,
            )
        )

    return papers


# ---------------------------------------------------------------------------
# (Optional) PDF text enrichment
# ---------------------------------------------------------------------------

def enrich_with_pdf_text(papers: List[Paper]) -> None:
    """
    Placeholder for PDF download + parsing.
    For now, we simply keep the abstract as the main text signal.
    If you later want full-text, plug in a PDF parser here (e.g., GROBID pipeline).
    """
    # In a more advanced version, you'd:
    #  - download PDF from paper.pdf_url
    #  - extract sections
    #  - append to or replace paper.abstract
    # For now: do nothing.
    pass


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def compute_embeddings(papers: List[Paper], model: SentenceTransformer) -> None:
    """
    Compute normalized embeddings for each paper (using its abstract).
    """
    texts = [p.abstract for p in papers]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    for p, emb in zip(papers, embeddings):
        p.embedding = emb


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def build_graph(papers: List[Paper], similarity_threshold: float = 0.4) -> nx.Graph:
    """
    Build a knowledge graph where:
      - Each paper is a node with attributes
      - Undirected edges connect papers whose cosine similarity >= similarity_threshold
    """
    G = nx.Graph()

    # Add paper nodes
    for p in papers:
        G.add_node(
            p.arxiv_id,
            type="paper",
            title=p.title,
            abstract=p.abstract,
            embedding=p.embedding,
        )

    # Add similarity-based edges
    n = len(papers)
    for i in range(n):
        for j in range(i + 1, n):
            pi = papers[i]
            pj = papers[j]
            if pi.embedding is None or pj.embedding is None:
                continue
            sim = float(np.dot(pi.embedding, pj.embedding))  # cosine because normalized
            if sim >= similarity_threshold:
                G.add_edge(
                    pi.arxiv_id,
                    pj.arxiv_id,
                    weight=sim,
                    type="similarity",
                )

    return G


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def explain_paper(
    arxiv_id: str,
    papers_by_id: Dict[str, Paper],
    G: nx.Graph,
    top_k_neighbors: int = 5,
) -> None:
    """
    Given an arxiv_id, print a short explanation:
      - Title + truncated abstract
      - Top-k most strongly connected neighbors in the graph
    This is now robust to missing papers / nodes (no StopIteration).
    """
    paper = papers_by_id.get(arxiv_id)
    if paper is None:
        print(f"[WARN] No paper with arxiv_id={arxiv_id} found in papers; skipping.\n")
        return

    print("=" * 80)
    print(f"Paper {arxiv_id}")
    print(f"Title: {paper.title}")
    print("-" * 80)
    print("Abstract (truncated):")
    print(
        textwrap.fill(
            paper.abstract[:600]
            + ("..." if len(paper.abstract) > 600 else ""),
            width=80,
        )
    )
    print("\nTop neighbors in the graph:")

    if arxiv_id not in G:
        print("  (This paper is not present as a node in the graph.)\n")
        return

    neighbors = []
    for nbr in G.neighbors(arxiv_id):
        edge_data = G.get_edge_data(arxiv_id, nbr) or {}
        w = edge_data.get("weight", 1.0)
        neighbors.append((nbr, w))

    if not neighbors:
        print("  (No neighbors in the graph for this paper.)\n")
        return

    neighbors.sort(key=lambda x: x[1], reverse=True)

    for nid, w in neighbors[:top_k_neighbors]:
        nbr_paper = papers_by_id.get(nid)
        title = nbr_paper.title if nbr_paper is not None else "<non-paper or missing>"
        print(f"  - {nid} (similarity={w:.3f})")
        print(f"    {title}")

    print()  # blank line at end


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # You can tweak this query as you like
    query = 'ti:"graph neural networks" AND cat:cs.LG'
    max_results = 10
    similarity_threshold = 0.4

    print("Fetching metadata from arXiv...")
    papers = fetch_papers(query=query, max_results=max_results)
    if not papers:
        print("No papers retrieved; exiting.")
        return

    print("Downloading PDFs and extracting text...")
    enrich_with_pdf_text(papers)

    print("Setting up NLP models...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Extracting concepts and computing embeddings...")
    compute_embeddings(papers, model=model)

    print("Linking references within the small corpus (similarity-based)...")
    print(f"  Using cosine similarity threshold: {similarity_threshold}")

    print("Building the graph...")
    G = build_graph(papers, similarity_threshold=similarity_threshold)

    # Index papers by arxiv_id for robust lookups (fixes StopIteration bug)
    papers_by_id: Dict[str, Paper] = {p.arxiv_id: p for p in papers}

    print("\nDone. Example queries:\n")

    # Only consider nodes that are known "paper" nodes and are present in papers_by_id
    paper_nodes = [
        n for n, data in G.nodes(data=True)
        if data.get("type") == "paper" and n in papers_by_id
    ]

    # Take up to 3 example papers
    for aid in paper_nodes[:3]:
        explain_paper(aid, papers_by_id, G)


if __name__ == "__main__":
    main()
