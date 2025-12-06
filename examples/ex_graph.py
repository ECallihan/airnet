from kg_ai_papers.config.settings import settings
from kg_ai_papers.ingest.arxiv_ingest import fetch_papers_and_pdfs
from kg_ai_papers.parsing.pipeline import parse_pdf_to_paper
from kg_ai_papers.nlp.concept_extraction import get_concept_extractor
from kg_ai_papers.nlp.embedding import get_embedding_model
from kg_ai_papers.graph.builder import build_graph
from kg_ai_papers.graph.storage import save_graph

# 1. Ingest papers (you’d implement this)
papers = fetch_papers_and_pdfs(["1706.03762", "1810.04805"])

# 2. Parse PDFs with Grobid into sections + references
for p in papers:
    p = parse_pdf_to_paper(p, pdf_path=p.pdf_path)

# 3. Concept extraction
ce = get_concept_extractor()
for p in papers:
    ce.extract_for_paper(p)

# 4. Embeddings (you’ll implement get_embedding_model similarly to concept extractor)
embedder = get_embedding_model()
for p in papers:
    p.embedding = embedder.encode_paper(p)

# 5. Citation map (you already have a prototype for this)
citation_map = build_citation_map_somehow(papers)

# 6. Graph
G = build_graph(papers, citation_map)
save_graph(G)
