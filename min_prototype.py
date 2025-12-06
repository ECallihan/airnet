import os
import re
import math
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import arxiv
import requests
from pypdf import PdfReader
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
import networkx as nx
from rapidfuzz import fuzz


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str
    text: str = ""
    concepts: List[Tuple[str, float]] = field(default_factory=list)
    embedding = None


# ----------------------------
# Config
# ----------------------------

# A small demo corpus – you can replace with any arXiv IDs you like
ARXIV_IDS = [
    "1706.03762",  # Attention is All You Need
    "1810.04805",  # BERT
    "2005.14165",  # RAG
]

MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformer model (lightweight)


# ----------------------------
# Step 1: Fetch metadata + PDFs from arxiv
# ----------------------------

def fetch_paper_metadata(arxiv_ids: List[str]) -> List[Paper]:
    id_query = " OR ".join(f"id:{aid}" for aid in arxiv_ids)
    search = arxiv.Search(
        query=id_query,
        max_results=len(arxiv_ids),
    )

    papers = []
    for result in search.results():
        papers.append(
            Paper(
                arxiv_id=result.get_short_id(),
                title=result.title,
                abstract=result.summary,
                pdf_url=result.pdf_url
            )
        )
    # Ensure order matches ARXIV_IDS
    id_to_paper = {p.arxiv_id: p for p in papers}
    return [id_to_paper[aid] for aid in arxiv_ids if aid in id_to_paper]


def download_pdf(pdf_url: str, dest_dir: str) -> str:
    r = requests.get(pdf_url)
    r.raise_for_status()
    filename = os.path.join(dest_dir, os.path.basename(pdf_url) + ".pdf")
    with open(filename, "wb") as f:
        f.write(r.content)
    return filename


# ----------------------------
# Step 2: Extract raw text from PDF
# ----------------------------

def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


# ----------------------------
# Step 3: Concept extraction with KeyBERT
# ----------------------------

def setup_nlp_models():
    kw_model = KeyBERT()
    st_model = SentenceTransformer(MODEL_NAME)
    return kw_model, st_model


def extract_concepts_for_paper(paper: Paper, kw_model: KeyBERT, top_n: int = 15) -> None:
    # Use abstract + maybe first part of text for concept extraction
    text = paper.abstract
    if len(paper.text) > 1000:
        text = paper.abstract + "\n\n" + paper.text[:5000]

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n
    )
    paper.concepts = keywords


def compute_paper_embedding(paper: Paper, st_model: SentenceTransformer) -> None:
    # Encode title + abstract as a simple representation
    text = paper.title + "\n\n" + paper.abstract
    paper.embedding = st_model.encode(text, convert_to_tensor=True)


# ----------------------------
# Step 4: Heuristic reference detection & linking
# ----------------------------

def extract_reference_block(text: str) -> str:
    """
    Very rough: look for 'References' or 'REFERENCES' and take everything after.
    """
    pattern = r"\bReferences\b|\bREFERENCES\b"
    match = re.search(pattern, text)
    if not match:
        return ""
    return text[match.start():]


def extract_reference_lines(ref_block: str) -> List[str]:
    """
    Another very rough heuristic: split lines, keep those that look like refs.
    """
    lines = [l.strip() for l in ref_block.splitlines()]
    # Keep non-empty lines that have a year-like pattern or numbering
    ref_lines = []
    for line in lines:
        if re.search(r"\b(19|20)\d{2}\b", line) or re.match(r"\[\d+\]", line):
            ref_lines.append(line)
    return ref_lines


def link_references_within_corpus(
    papers: List[Paper]
) -> Dict[str, List[str]]:
    """
    For each paper, try to match reference lines to titles of other papers
    in our small corpus using fuzzy matching.
    Returns: { citing_arxiv_id: [cited_arxiv_id, ...], ... }
    """
    id_to_title = {p.arxiv_id: p.title for p in papers}
    id_to_text = {p.arxiv_id: p.text for p in papers}

    citation_map: Dict[str, List[str]] = {p.arxiv_id: [] for p in papers}

    for paper in papers:
        ref_block = extract_reference_block(paper.text)
        if not ref_block:
            continue
        ref_lines = extract_reference_lines(ref_block)

        for line in ref_lines:
            best_match_id = None
            best_score = 0
            for other_id, other_title in id_to_title.items():
                if other_id == paper.arxiv_id:
                    continue
                score = fuzz.partial_ratio(other_title.lower(), line.lower())
                if score > best_score:
                    best_score = score
                    best_match_id = other_id

            # Only accept high-confidence matches
            if best_match_id and best_score >= 80:
                if best_match_id not in citation_map[paper.arxiv_id]:
                    citation_map[paper.arxiv_id].append(best_match_id)

    return citation_map


# ----------------------------
# Step 5: Build the graph
# ----------------------------

def build_graph(papers: List[Paper], citation_map: Dict[str, List[str]]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # Add paper nodes
    for p in papers:
        G.add_node(
            f"paper:{p.arxiv_id}",
            type="paper",
            arxiv_id=p.arxiv_id,
            title=p.title
        )

    # Add concept nodes and PAPER_HAS_CONCEPT edges
    concept_to_id = {}
    for p in papers:
        for concept, score in p.concepts:
            c_label = concept.strip()
            if not c_label:
                continue
            cid = concept_to_id.get(c_label)
            if cid is None:
                cid = f"concept:{len(concept_to_id)}"
                concept_to_id[c_label] = cid
                G.add_node(cid, type="concept", label=c_label)
            G.add_edge(
                f"paper:{p.arxiv_id}",
                cid,
                type="PAPER_HAS_CONCEPT",
                weight=float(score)
            )

    # Add citation edges + influence scores based on cosine similarity
    for citing_id, cited_ids in citation_map.items():
        citing_paper = next(p for p in papers if p.arxiv_id == citing_id)
        for cited_id in cited_ids:
            cited_paper = next(p for p in papers if p.arxiv_id == cited_id)
            if citing_paper.embedding is None or cited_paper.embedding is None:
                continue
            sim = util.cos_sim(citing_paper.embedding, cited_paper.embedding).item()
            # Normalize to [0,1] just in case
            influence_score = max(0.0, min(1.0, (sim + 1.0) / 2.0))
            G.add_edge(
                f"paper:{citing_id}",
                f"paper:{cited_id}",
                type="PAPER_CITES_PAPER",
                influence_score=influence_score,
                similarity=sim
            )

    return G


# ----------------------------
# Step 6: Query function
# ----------------------------

def explain_paper(
    arxiv_id: str,
    papers: List[Paper],
    G: nx.MultiDiGraph,
    top_k_concepts: int = 10,
    top_k_influential_refs: int = 5,
    top_k_influenced_by: int = 5
):
    node_id = f"paper:{arxiv_id}"
    # paper = next(p for p in papers if p.arxiv_id == arxiv_id)
    paper = next((p for p in papers if p.arxiv_id == arxiv_id), None)
    if paper is None:
        print(f"[WARN] No paper with arxiv_id={arxiv_id} found in papers; skipping.")
        return
    print("=" * 80)
    print(f"Paper: {paper.title} (arXiv:{paper.arxiv_id})")
    print("=" * 80)
    print("\nTop concepts:")
    for concept, score in paper.concepts[:top_k_concepts]:
        print(f"  - {concept}  (score={score:.3f})")

    # Outgoing citations (papers this one references)
    out_edges = [
        (u, v, d) for u, v, d in G.out_edges(node_id, data=True)
        if d.get("type") == "PAPER_CITES_PAPER"
    ]
    out_edges_sorted = sorted(
        out_edges,
        key=lambda x: x[2].get("influence_score", 0.0),
        reverse=True
    )

    print("\nMost influential referenced papers (this paper builds on):")
    if not out_edges_sorted:
        print("  (No matched references within this small corpus.)")
    for u, v, d in out_edges_sorted[:top_k_influential_refs]:
        cited_title = G.nodes[v]["title"]
        print(f"  -> {cited_title}  [influence={d['influence_score']:.3f}]")

    # Incoming citations (papers that reference this one)
    in_edges = [
        (u, v, d) for u, v, d in G.in_edges(node_id, data=True)
        if d.get("type") == "PAPER_CITES_PAPER"
    ]
    in_edges_sorted = sorted(
        in_edges,
        key=lambda x: x[2].get("influence_score", 0.0),
        reverse=True
    )

    print("\nPapers most influenced by this paper (within corpus):")
    if not in_edges_sorted:
        print("  (No citing papers within this small corpus.)")
    for u, v, d in in_edges_sorted[:top_k_influenced_by]:
        citing_title = G.nodes[u]["title"]
        print(f"  <- {citing_title}  [influence={d['influence_score']:.3f}]")


# ----------------------------
# Main driver
# ----------------------------

def main():
    print("Fetching metadata from arXiv...")
    papers = fetch_paper_metadata(ARXIV_IDS)

    print("Downloading PDFs and extracting text...")
    with tempfile.TemporaryDirectory() as tmpdir:
        for p in papers:
            print(f"  - {p.arxiv_id}: downloading PDF...")
            pdf_path = download_pdf(p.pdf_url, tmpdir)
            print(f"    extracting text...")
            p.text = extract_pdf_text(pdf_path)

    print("Setting up NLP models...")
    kw_model, st_model = setup_nlp_models()

    print("Extracting concepts and computing embeddings...")
    for p in papers:
        print(f"  - {p.arxiv_id}: {p.title}")
        extract_concepts_for_paper(p, kw_model, top_n=20)
        compute_paper_embedding(p, st_model)

    print("Linking references within the small corpus...")
    citation_map = link_references_within_corpus(papers)

    print("Building the graph...")
    G = build_graph(papers, citation_map)

    print("\nDone. Example queries:\n")
    for aid in ARXIV_IDS:
        explain_paper(aid, papers, G)
        print("\n\n")


if __name__ == "__main__":
    main()
