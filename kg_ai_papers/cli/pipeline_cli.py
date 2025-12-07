# kg_ai_papers/cli/pipeline_cli.py

from __future__ import annotations

from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress

from kg_ai_papers.config.settings import settings
from kg_ai_papers.graph.builder import build_graph
from kg_ai_papers.graph.storage import load_graph, save_graph
from kg_ai_papers.ingest.arxiv_ingest import ingest_arxiv_ids  # assumes this exists
from kg_ai_papers.parsing.pipeline import ensure_enriched

app = typer.Typer(help="End-to-end pipeline: ingest -> parse -> enrich -> graph.")
console = Console()


@app.command("run")
def run_pipeline(
    arxiv_ids: List[str] = typer.Argument(
        ...,
        help="One or more arXiv IDs to ingest and add to the graph.",
    ),
    graph_name: Optional[str] = typer.Option(
        None,
        "--graph",
        "-g",
        help="Logical graph name (without extension). Defaults to settings.GRAPH_DEFAULT_NAME.",
    ),
    overwrite_pdfs: bool = typer.Option(
        False,
        "--overwrite-pdfs",
        help="Re-download PDFs even if they already exist.",
    ),
    overwrite_enriched: bool = typer.Option(
        False,
        "--overwrite-enriched",
        help="Re-run Grobid + NLP even if enriched JSON already exists.",
    ),
):
    """
    Run the full pipeline for the given arXiv IDs and update the stored graph.
    """
    graph_name = graph_name or settings.GRAPH_DEFAULT_NAME

    console.rule("[bold cyan]AI Paper KG Pipeline[/bold cyan]")

    # ------------------------------------------------------------------
    # 1. Ingest PDFs + metadata
    # ------------------------------------------------------------------

    console.print("[bold]Ingesting papers from arXiv...[/bold]")

    papers = ingest_arxiv_ids(
        arxiv_ids,
        overwrite_pdfs=overwrite_pdfs,
    )

    if not papers:
        console.print("[yellow]No papers ingested.[/yellow]")
        raise typer.Exit(code=0)

    # ------------------------------------------------------------------
    # 2. Parse PDFs with Grobid + enrich (concepts + embedding)
    #    using per-paper enriched JSON to avoid recomputation.
    # ------------------------------------------------------------------

    enriched_papers = []
    with Progress() as progress:
        t_parse = progress.add_task(
            "Parsing + enriching papers...", total=len(papers)
        )
        for p in papers:
            # ensure_enriched handles:
            #  - skip if enriched JSON exists (unless overwrite_enriched=True)
            #  - Grobid + concept extraction + embedding
            p_enriched = ensure_enriched(
                p,
                pdf_path=p.pdf_path,
                overwrite=overwrite_enriched,
            )
            enriched_papers.append(p_enriched)
            progress.advance(t_parse)

    # ------------------------------------------------------------------
    # 3. Load existing graph (if any) and incrementally update
    # ------------------------------------------------------------------

    console.print("[bold]Loading existing graph (if any) and updating...[/bold]")

    try:
        G = load_graph(graph_name)
    except FileNotFoundError:
        G = None

    G = build_graph(enriched_papers, existing_graph=G)
    path = save_graph(G, graph_name)

    console.print(
        f"[green]Pipeline complete! Graph saved to: [bold]{path}[/bold][/green]"
    )
