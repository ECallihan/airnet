# kg_ai_papers/cli/query_cli.py

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from kg_ai_papers.graph.schema import NodeType, EdgeType
from kg_ai_papers.graph.storage import load_graph
from kg_ai_papers.api.query import (
    get_paper_influence_view,
    get_paper_concepts,
)

app = typer.Typer(help="Query the AI paper knowledge graph.")
console = Console()


def _load_graph_by_name(graph_name: Optional[str]):
    """
    Helper: load the graph (optionally by name).
    """
    if graph_name:
        return load_graph(graph_name)
    return load_graph()

def _find_paper_node(G, paper_id: str) -> Optional[str]:
    """
    Find the node id for a paper given its arxiv_id.
    """
    for node, data in G.nodes(data=True):
        if data.get("type") == NodeType.PAPER.value and data.get("arxiv_id") == paper_id:
            return node
    return None


def _concept_nodes_by_label(G) -> dict:
    """
    Build a mapping from concept label -> list of concept node ids.
    """
    mapping = {}
    for node, data in G.nodes(data=True):
        if data.get("type") == NodeType.CONCEPT.value:
            label = data.get("label")
            if not label:
                continue
            mapping.setdefault(label, []).append(node)
    return mapping


def _concept_neighborhood(G, concept_label: str, current_paper_node: str, max_neighbors: int = 3):
    """
    For a given concept, find other papers that have a strong association with it.
    Returns a list of dicts: {arxiv_id, title, weight}.
    """
    concept_map = _concept_nodes_by_label(G)
    concept_nodes = concept_map.get(concept_label, [])
    neighbors = {}

    for c_node in concept_nodes:
        # Incoming edges: paper -> concept
        for paper_node, _, data in G.in_edges(c_node, data=True):
            if data.get("type") != EdgeType.PAPER_HAS_CONCEPT.value:
                continue
            if paper_node == current_paper_node:
                continue
            w = float(data.get("weight", 0.0))
            paper_data = G.nodes[paper_node]
            arxiv_id = paper_data.get("arxiv_id", "")
            title = paper_data.get("title", "")
            # Keep max weight per neighbor
            if arxiv_id not in neighbors or w > neighbors[arxiv_id]["weight"]:
                neighbors[arxiv_id] = {"arxiv_id": arxiv_id, "title": title, "weight": w}

    # Sort by descending weight
    ordered = sorted(neighbors.values(), key=lambda x: x["weight"], reverse=True)
    return ordered[:max_neighbors]


def _concept_lineage_from_references(G, concept_label: str, current_paper_node: str, max_refs: int = 3):
    """
    Among the references of the current paper, find those that also have this concept.
    Returns a list of dicts: {arxiv_id, title, weight}.
    """
    concept_map = _concept_nodes_by_label(G)
    concept_nodes = concept_map.get(concept_label, [])
    if not concept_nodes:
        return []

    lineage = {}

    # Edges current_paper_node -> ref_node with type PAPER_CITES_PAPER
    for _, ref_node, data in G.out_edges(current_paper_node, data=True):
        if data.get("type") != EdgeType.PAPER_CITES_PAPER.value:
            continue

        # Does ref_node have this concept?
        for _, c_node, d2 in G.out_edges(ref_node, data=True):
            if d2.get("type") != EdgeType.PAPER_HAS_CONCEPT.value:
                continue
            if c_node not in concept_nodes:
                continue

            w = float(d2.get("weight", 0.0))
            ref_data = G.nodes[ref_node]
            arxiv_id = ref_data.get("arxiv_id", "")
            title = ref_data.get("title", "")
            if arxiv_id not in lineage or w > lineage[arxiv_id]["weight"]:
                lineage[arxiv_id] = {"arxiv_id": arxiv_id, "title": title, "weight": w}

    ordered = sorted(lineage.values(), key=lambda x: x["weight"], reverse=True)
    return ordered[:max_refs]


# ---------------------------------------------------------------------------
# list: show all papers in the graph
# ---------------------------------------------------------------------------

@app.command("list")
def list_papers(
    graph: Optional[str] = typer.Option(
        None,
        "--graph",
        "-g",
        help="Graph name to load (without .gpickle). Defaults to settings.GRAPH_DEFAULT_NAME.",
    )
):
    """
    List all papers currently in the graph.
    """
    G = _load_graph_by_name(graph)

    rows = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "paper":
            rows.append(
                (
                    data.get("arxiv_id", ""),
                    data.get("title", ""),
                )
            )

    if not rows:
        console.print("[yellow]No papers found in graph.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="Papers in Knowledge Graph")
    table.add_column("arxiv_id", style="cyan", no_wrap=True)
    table.add_column("title", style="white")

    for arxiv_id, title in sorted(rows):
        table.add_row(arxiv_id, title)

    console.print(table)


# ---------------------------------------------------------------------------
# paper: full influence view (concepts + references + influenced)
# ---------------------------------------------------------------------------

@app.command("paper")
def paper_view(
    paper_id: str = typer.Argument(..., help="arXiv ID of the paper (e.g. 1706.03762 or 1706.03762v7)."),
    graph: Optional[str] = typer.Option(
        None,
        "--graph",
        "-g",
        help="Graph name to load (without .gpickle). Defaults to settings.GRAPH_DEFAULT_NAME.",
    ),
):
    """
    Show full view of a paper: basic info, concepts, influential references, influenced papers.
    """
    G = _load_graph_by_name(graph)

    try:
        view = get_paper_influence_view(G, paper_id)
    except KeyError:
        console.print(f"[red]Paper {paper_id!r} not found in graph.[/red]")
        raise typer.Exit(code=1)

    # Header
    console.rule(f"[bold cyan]{view.paper.arxiv_id}[/bold cyan]")
    console.print(f"[bold]{view.paper.title}[/bold]")
    if view.paper.abstract:
        console.print()
        console.print("[bold]Abstract:[/bold]")
        console.print(view.paper.abstract)

    # Concepts
    console.print()
    console.print("[bold magenta]Concepts[/bold magenta]")
    if not view.concepts:
        console.print("[yellow]No concepts found.[/yellow]")
    else:
        t_concepts = Table(show_header=True, header_style="bold magenta")
        t_concepts.add_column("Concept", style="cyan")
        t_concepts.add_column("Weight", style="white", justify="right")
        for c in view.concepts:
            t_concepts.add_row(c.label, f"{c.weight:.3f}")
        console.print(t_concepts)

    # Influential references
    console.print()
    console.print("[bold magenta]Influential References[/bold magenta]")
    if not view.influential_references:
        console.print("[yellow]No influential references in graph.[/yellow]")
    else:
        t_refs = Table(show_header=True, header_style="bold magenta")
        t_refs.add_column("arxiv_id", style="cyan", no_wrap=True)
        t_refs.add_column("Title", style="white")
        t_refs.add_column("Influence", justify="right")
        t_refs.add_column("Similarity", justify="right")
        for r in view.influential_references:
            t_refs.add_row(
                r.arxiv_id,
                r.title or "",
                f"{r.influence_score:.3f}" if r.influence_score is not None else "",
                f"{r.similarity:.3f}" if r.similarity is not None else "",
            )
        console.print(t_refs)

    # Influenced papers
    console.print()
    console.print("[bold magenta]Papers Influenced by This Paper[/bold magenta]")
    if not view.influenced_papers:
        console.print("[yellow]No influenced papers in graph.[/yellow]")
    else:
        t_inf = Table(show_header=True, header_style="bold magenta")
        t_inf.add_column("arxiv_id", style="cyan", no_wrap=True)
        t_inf.add_column("Title", style="white")
        t_inf.add_column("Influence", justify="right")
        t_inf.add_column("Similarity", justify="right")
        for r in view.influenced_papers:
            t_inf.add_row(
                r.arxiv_id,
                r.title or "",
                f"{r.influence_score:.3f}" if r.influence_score is not None else "",
                f"{r.similarity:.3f}" if r.similarity is not None else "",
            )
        console.print(t_inf)


# ---------------------------------------------------------------------------
# concepts: only show the concepts for a given paper
# ---------------------------------------------------------------------------

@app.command("concepts")
def concepts(
    paper_id: str = typer.Argument(..., help="arXiv ID of the paper (e.g. 1706.03762 or 1706.03762v7)."),
    graph: Optional[str] = typer.Option(
        None,
        "--graph",
        "-g",
        help="Graph name to load (without .gpickle). Defaults to settings.GRAPH_DEFAULT_NAME.",
    ),
    max_concepts: int = typer.Option(
        10,
        "--top",
        "-t",
        help="Number of top concepts to display.",
    ),
):
    """
    Rich concept view for a given paper:
    - Top concepts
    - For each concept: neighborhood (other papers) + lineage (which references also use it)
    """
    G = _load_graph_by_name(graph)

    # Use full view to get title + concepts
    try:
        full_view = get_paper_influence_view(G, paper_id)
    except KeyError:
        console.print(f"[red]Paper {paper_id!r} not found in graph.[/red]")
        raise typer.Exit(code=1)

    title = full_view.paper.title or paper_id
    real_id = full_view.paper.arxiv_id or paper_id

    # Find node id in graph
    paper_node = _find_paper_node(G, real_id)
    if paper_node is None:
        console.print(f"[red]Paper node for {real_id!r} not found in graph.[/red]")
        raise typer.Exit(code=1)

    concept_models = full_view.concepts[:max_concepts]

    # Header panel
    console.rule(f"[bold cyan]Concepts for {title}[/bold cyan]")
    console.print(f"[dim]arXiv: {real_id}[/dim]\n")

    if not concept_models:
        console.print("[yellow]No concepts found for this paper.[/yellow]")
        raise typer.Exit(code=0)

    # Overall concepts table
    t_concepts = Table(show_header=True, header_style="bold magenta", title="Top Concepts")
    t_concepts.add_column("#", style="dim", justify="right")
    t_concepts.add_column("Concept", style="cyan")
    t_concepts.add_column("Weight", style="white", justify="right")

    for idx, c in enumerate(concept_models, start=1):
        t_concepts.add_row(str(idx), c.label, f"{c.weight:.3f}")

    console.print(t_concepts)
    console.print()

    # Detailed per-concept view
    for idx, c in enumerate(concept_models, start=1):
        console.rule(f"[bold green]#{idx} {c.label}[/bold green]")

        # Neighborhood: other papers sharing this concept
        neighbors = _concept_neighborhood(G, c.label, paper_node, max_neighbors=3)
        lineage = _concept_lineage_from_references(G, c.label, paper_node, max_refs=3)

        # Concept summary
        console.print(f"[bold]Weight:[/bold] {c.weight:.3f}")
        console.print()

        # Neighborhood table
        console.print("[bold magenta]Concept Neighborhood (other papers)[/bold magenta]")
        if not neighbors:
            console.print("[dim]No other papers with this concept found in the graph.[/dim]")
        else:
            t_neigh = Table(show_header=True, header_style="bold magenta")
            t_neigh.add_column("arxiv_id", style="cyan", no_wrap=True)
            t_neigh.add_column("Title", style="white")
            t_neigh.add_column("Weight", style="white", justify="right")
            for n in neighbors:
                t_neigh.add_row(
                    n["arxiv_id"] or "",
                    n["title"] or "",
                    f"{n['weight']:.3f}",
                )
            console.print(t_neigh)

        console.print()

        # Lineage table
        console.print("[bold magenta]Concept Lineage (among this paper's references)[/bold magenta]")
        if not lineage:
            console.print("[dim]None of this paper's references share this concept (in current graph).[/dim]")
        else:
            t_lin = Table(show_header=True, header_style="bold magenta")
            t_lin.add_column("arxiv_id", style="cyan", no_wrap=True)
            t_lin.add_column("Title", style="white")
            t_lin.add_column("Weight", style="white", justify="right")
            for n in lineage:
                t_lin.add_row(
                    n["arxiv_id"] or "",
                    n["title"] or "",
                    f"{n['weight']:.3f}",
                )
            console.print(t_lin)

        console.print()
