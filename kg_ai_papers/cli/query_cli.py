# kg_ai_papers/cli/query_cli.py

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from kg_ai_papers.api.query import explain_paper

app = typer.Typer(help="Query the AI paper knowledge graph.")
console = Console()


@app.command("paper")
def paper_command(
    arxiv_id: str = typer.Argument(..., help="arXiv ID of the focal paper."),
    graph_name: str = typer.Option(
        "graph",
        "--graph-name",
        "-g",
        help="Name of the graph file (without extension) to load.",
    ),
    top_k_concepts: int = typer.Option(
        10, "--top-concepts", "-c", help="Number of top concepts to display."
    ),
    top_k_references: int = typer.Option(
        5, "--top-refs", "-r", help="Number of top influential references to display."
    ),
    top_k_influenced: int = typer.Option(
        5, "--top-influenced", "-i", help="Number of top influenced papers to display."
    ),
):
    """
    Explain a paper: show its main concepts, the papers it builds on,
    and the papers that build on it.
    """
    try:
        result = explain_paper(
            arxiv_id=arxiv_id,
            graph_name=graph_name,
            top_k_concepts=top_k_concepts,
            top_k_references=top_k_references,
            top_k_influenced=top_k_influenced,
        )
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    # Header panel
    title_text = Text(result.paper.title, style="bold cyan")
    subtitle = f"arXiv:{result.paper.arxiv_id}"
    header = Panel(title_text, subtitle=subtitle, expand=True)
    console.print(header)
    console.print()

    # Abstract (truncated)
    if result.paper.abstract:
        abstract = result.paper.abstract.strip()
        if len(abstract) > 800:
            abstract = abstract[:800] + " …"
        console.print(Panel(abstract, title="Abstract", expand=True))
        console.print()

    # Concepts table
    if result.concepts:
        table = Table(title="Top Concepts", show_lines=False)
        table.add_column("#", style="dim", width=4)
        table.add_column("Concept")
        table.add_column("Weight", justify="right")

        for idx, concept in enumerate(result.concepts, start=1):
            table.add_row(str(idx), concept.label, f"{concept.weight:.3f}")

        console.print(table)
        console.print()
    else:
        console.print("[yellow]No concepts found for this paper.[/yellow]")
        console.print()

    # Influential references
    if result.influential_references:
        refs_table = Table(title="Most Influential References (papers this builds on)")
        refs_table.add_column("#", style="dim", width=4)
        refs_table.add_column("arXiv ID", style="magenta")
        refs_table.add_column("Title")
        refs_table.add_column("Influence", justify="right")
        refs_table.add_column("Cosine Sim", justify="right")

        for idx, ref in enumerate(result.influential_references, start=1):
            refs_table.add_row(
                str(idx),
                ref.arxiv_id,
                ref.title,
                f"{ref.influence_score:.3f}",
                f"{ref.similarity:.3f}",
            )

        console.print(refs_table)
        console.print()
    else:
        console.print(
            "[yellow]No influential references found within this graph (maybe small corpus?).[/yellow]"
        )
        console.print()

    # Influenced papers
    if result.influenced_papers:
        inf_table = Table(title="Papers Most Influenced by This One")
        inf_table.add_column("#", style="dim", width=4)
        inf_table.add_column("arXiv ID", style="magenta")
        inf_table.add_column("Title")
        inf_table.add_column("Influence", justify="right")
        inf_table.add_column("Cosine Sim", justify="right")

        for idx, inf in enumerate(result.influenced_papers, start=1):
            inf_table.add_row(
                str(idx),
                inf.arxiv_id,
                inf.title,
                f"{inf.influence_score:.3f}",
                f"{inf.similarity:.3f}",
            )

        console.print(inf_table)
    else:
        console.print(
            "[yellow]No influenced papers found within this graph (maybe small corpus?).[/yellow]"
        )


if __name__ == "__main__":
    app()
