# kg_ai_papers/cli/pipeline_cli.py

from __future__ import annotations

from typing import List, Dict

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rapidfuzz import fuzz

from kg_ai_papers.config.settings import settings
from kg_ai_papers.ingest.arxiv_ingest import fetch_papers_and_pdfs
from kg_ai_papers.parsing.pipeline import parse_pdf_to_paper
from kg_ai_papers.nlp.concept_extraction import get_concept_extractor
from kg_ai_papers.nlp.embedding import get_embedding_model
from kg_ai_papers.graph.builder import build_graph
from kg_ai_papers.graph.storage import save_graph
from kg_ai_papers.models.paper import Paper

app = typer.Typer(help="End-to-end pipeline for the AI paper knowledge graph.")
console = Console()


# ---------------------------------------------------------------------------
# Citation map helper: match Grobid references to our in-memory corpus
# ---------------------------------------------------------------------------

def build_citation_map_from_references(papers: List[Paper]) -> Dict[str, List[str]]:
    """
    Build a citation map:

        citing_arxiv_id -> [cited_arxiv_id, ...]

    using Grobid-parsed references and fuzzy title matching against our Paper list.
    """
    citation_map: Dict[str, List[str]] = {p.arxiv_id: [] for p in papers}

    # Map: title_lower -> arxiv_id (simple map; we also fuzzy match)
    id_to_title = {p.arxiv_id: p.title for p in papers}

    for p in papers:
        if not p.references:
            continue

        for ref in p.references:
            if not ref.title:
                continue
            ref_title_lower = ref.title.strip().lower()

            best_id = None
            best_score = 0.0

            for other_id, other_title in id_to_title.items():
                if other_id == p.arxiv_id:
                    continue
                score = fuzz.partial_ratio(other_title.lower(), ref_title_lower)
                if score > best_score:
                    best_score = score
                    best_id = other_id

            # Only accept matches above a high threshold
            if best_id and best_score >= 85:
                if best_id not in citation_map[p.arxiv_id]:
                    citation_map[p.arxiv_id].append(best_id)

    return citation_map


# ---------------------------------------------------------------------------
# Main pipeline command
# ---------------------------------------------------------------------------

@app.command("run")
def run_pipeline(
    arxiv_ids: List[str] = typer.Argument(
        ...,
        help="One or more arXiv IDs to ingest and add to the knowledge graph.",
    ),
    graph_name: str = typer.Option(
        "graph",
        "--graph-name",
        "-g",
        help="Name of the graph file to save under data/graph/{name}.gpickle",
    ),
    overwrite_pdfs: bool = typer.Option(
        False,
        "--overwrite-pdfs",
        help="Force re-download of PDFs even if they already exist.",
    ),
):
    """
    Run the full pipeline for the given arXiv IDs:

      ingest -> Grobid parse -> concept extraction -> embeddings -> graph build
    """
    console.rule("[bold cyan]AI Paper KG Pipeline[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # ---------------------------------------------------------------
        # 1. Ingest arXiv metadata + PDFs
        # ---------------------------------------------------------------
        t_ingest = progress.add_task("Ingesting papers from arXiv...", total=None)
        papers = fetch_papers_and_pdfs(arxiv_ids, overwrite_pdfs=overwrite_pdfs)
        progress.update(t_ingest, completed=1)
        progress.remove_task(t_ingest)

        if not papers:
            console.print("[red]No papers ingested. Exiting.[/red]")
            raise typer.Exit(code=1)

        # ---------------------------------------------------------------
        # 2. Grobid parsing -> sections + references
        # ---------------------------------------------------------------
        t_parse = progress.add_task(
            "Parsing PDFs with Grobid...", total=len(papers)
        )
        for p in papers:
            parse_pdf_to_paper(p, pdf_path=p.pdf_path)
            progress.advance(t_parse)
        progress.remove_task(t_parse)

        # ---------------------------------------------------------------
        # 3. Concept extraction
        # ---------------------------------------------------------------
        ce = get_concept_extractor()
        t_concepts = progress.add_task(
            "Extracting concepts...", total=len(papers)
        )
        for p in papers:
            ce.extract_for_paper(p)
            progress.advance(t_concepts)
        progress.remove_task(t_concepts)

        # ---------------------------------------------------------------
        # 4. Embeddings (batch)
        # ---------------------------------------------------------------
        embedder = get_embedding_model()
        t_emb = progress.add_task(
            "Computing embeddings...", total=None
        )
        # One batch call; internally sets p.embedding
        embedder.encode_papers_batch(papers)
        progress.update(t_emb, completed=1)
        progress.remove_task(t_emb)

        # ---------------------------------------------------------------
        # 5. Citation map
        # ---------------------------------------------------------------
        t_cmap = progress.add_task(
            "Building citation map from references...", total=None
        )
        citation_map = build_citation_map_from_references(papers)
        progress.update(t_cmap, completed=1)
        progress.remove_task(t_cmap)

        # ---------------------------------------------------------------
        # 6. Graph build + save
        # ---------------------------------------------------------------
        t_graph = progress.add_task("Building and saving graph...", total=None)
        G = build_graph(papers, citation_map)
        path = save_graph(G, name=graph_name)
        progress.update(t_graph, completed=1)
        progress.remove_task(t_graph)

    console.print()
    console.print(
        f"[bold green]Pipeline complete![/bold green] Graph saved to: [cyan]{path}[/cyan]"
    )


if __name__ == "__main__":
    app()
