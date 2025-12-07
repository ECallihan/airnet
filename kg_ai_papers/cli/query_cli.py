from pathlib import Path
import pickle

import networkx as nx
import typer
from rich import print

from rich.console import Console
from rich.table import Table

from kg_ai_papers.api.query import (
    get_paper_concepts,
    get_paper_influence_view,
)
from kg_ai_papers.models.paper import Paper


GRAPH_PATH = Path("data/graph/graph.gpickle")


def _load_graph():
    """
    Load the knowledge graph from disk if it exists.
    Returns a networkx.MultiDiGraph or None.
    """
    if not GRAPH_PATH.exists():
        return None

    with GRAPH_PATH.open("rb") as f:
        G = pickle.load(f)
    return G



app = typer.Typer(help="Query the AI paper knowledge graph.")


# ----------------------------------------------------------------------
# LIST COMMAND
# ----------------------------------------------------------------------
@app.command("list")
def list_papers():
    """List all papers currently in the graph."""
    G = _load_graph()
    if G is None or G.number_of_nodes() == 0:
        print("[red]No papers found in the graph yet.[/red]")
        return

    rows = []

    # Heuristic: treat any node that has a "title" as a paper node.
    # Prefer the explicit `arxiv_id` attribute if present; otherwise use the node id.
    for node_id, data in G.nodes(data=True):
        title = data.get("title")
        if not title:
            continue  # likely a concept or other node type

        arxiv_id = data.get("arxiv_id", node_id)
        rows.append((str(arxiv_id), str(title)))

    if not rows:
        print("[red]No papers found in the graph yet.[/red]")
        return

    # Sort by arxiv_id for nicer output
    rows.sort(key=lambda r: r[0])

    table = Table(title="Papers in Knowledge Graph")
    table.add_column("arXiv ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")

    for arxiv_id, title in rows:
        table.add_row(arxiv_id, title)

    console = Console()
    console.print(table)


# ----------------------------------------------------------------------
# PAPER COMMAND (FULL DETAILS)
# ----------------------------------------------------------------------
@app.command("paper")
def paper(
    arxiv_id: str = typer.Argument(..., help="arXiv id of the paper"),
    top: int = typer.Option(10, "--top", help="Top N concepts/references/influenced papers to show"),
):
    """Show a high-level influence view for a single paper."""
    G = _load_graph()
    if G is None:
        print("[red]No graph found. Run the pipeline first.[/red]")
        raise typer.Exit(code=1)

    try:
        view = get_paper_influence_view(
            G,
            arxiv_id=arxiv_id,
            top_k_concepts=top,
            top_k_references=top,
            top_k_influenced=top,
        )
    except Exception as e:
        print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    # Example pretty-print (adjust if you already have something):
    console = Console()
    console.print(f"[bold]Paper:[/bold] {view.paper.arxiv_id} — {view.paper.title}")

    console.print("\n[bold]Top concepts:[/bold]")
    for c in view.concepts:
        console.print(f"  • {c.label} (weight={c.weight:.3f})")

    console.print("\n[bold]Most influential references:[/bold]")
    for r in view.references:
        console.print(
            f"  • {r.arxiv_id} — {r.title} "
            f"(influence_score={r.influence_score:.3f}, similarity={r.similarity:.3f})"
        )

    console.print("\n[bold]Papers influenced by this one:[/bold]")
    if not view.influenced:
        console.print("  (none)")
    else:
        for r in view.influenced:
            console.print(
                f"  • {r.arxiv_id} — {r.title} "
                f"(influence_score={r.influence_score:.3f}, similarity={r.similarity:.3f})"
            )


# ----------------------------------------------------------------------
# CONCEPTS COMMAND
# ----------------------------------------------------------------------
@app.command("concepts")
def concepts(
    arxiv_id: str = typer.Argument(..., help="arXiv id of the paper"),
    top: int = typer.Option(10, "--top", help="Top N concepts to show"),
):
    """Show the top concepts for a given paper."""
    G = _load_graph()
    if G is None:
        print("[red]No graph found in data/graph yet. Run the pipeline first.[/red]")
        raise typer.Exit(code=1)

    try:
        concepts = get_paper_concepts(G, arxiv_id, top)
    except Exception as e:
        print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console = Console()
    console.print(f"[bold]Top concepts for {arxiv_id}[/bold]")
    if not concepts:
        console.print("  (no concepts found)")
        return

    for c in concepts:
        console.print(f"  • {c.label} (weight={c.weight:.3f})")


# ----------------------------------------------------------------------
# INFLUENCE COMMAND
# ----------------------------------------------------------------------
@app.command("influence")
def influence(
    arxiv_id: str = typer.Argument(..., help="arXiv id of the paper"),
    top: int = typer.Option(10, "--top", help="Top N concepts/references/influenced papers to show"),
):
    """Show the influence view for a paper (concepts + refs + influenced papers)."""
    G = _load_graph()
    if G is None:
        print("[red]No graph found. Run the pipeline first.[/red]")
        raise typer.Exit(code=1)

    try:
        view = get_paper_influence_view(
            G,
            arxiv_id=arxiv_id,
            top_k_concepts=top,
            top_k_references=top,
            top_k_influenced=top,
        )
    except Exception as e:
        print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console = Console()
    console.print(f"[bold]Paper:[/bold] {view.paper.arxiv_id} — {view.paper.title}")

    console.print("\n[bold]Top concepts:[/bold]")
    for c in view.concepts:
        console.print(f"  • {c.label} (weight={c.weight:.3f})")

    console.print("\n[bold]Most influential references:[/bold]")
    for r in view.references:
        console.print(
            f"  • {r.arxiv_id} — {r.title} "
            f"(influence_score={r.influence_score:.3f}, similarity={r.similarity:.3f})"
        )

    console.print("\n[bold]Papers influenced by this one:[/bold]")
    if not view.influenced:
        console.print("  (none)")
    else:
        for r in view.influenced:
            console.print(
                f"  • {r.arxiv_id} — {r.title} "
                f"(influence_score={r.influence_score:.3f}, similarity={r.similarity:.3f})"
            )