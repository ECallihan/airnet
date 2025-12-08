from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import typer
from rich.console import Console
from rich.table import Table

from kg_ai_papers.api.query import get_paper_influence_view
from kg_ai_papers.graph.builder import paper_node_id
from kg_ai_papers.graph.io import load_graph as load_graph_file
from kg_ai_papers.graph.storage import load_latest_graph
from kg_ai_papers.config.settings import settings

app = typer.Typer(
    help="Read/query utilities over a saved AirNet knowledge graph."
)

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_graph(graph_file: Optional[Path]) -> nx.MultiDiGraph:
    """
    Load a saved graph.

    If --graph-file is provided, that exact file is loaded.
    Otherwise, we try to load the 'latest' graph from settings.graph_dir.
    """
    if graph_file is not None:
        path = Path(graph_file)
        if not path.exists():
            console.print(f"[red]Graph file not found:[/red] {path}")
            raise typer.Exit(code=1)
        return load_graph_file(path)  # type: ignore[return-value]

    # Fallback: use the latest graph in the configured graph directory.
    G = load_latest_graph(settings.graph_dir)
    if G is None:
        console.print(
            f"[red]No graph found in {settings.graph_dir}.[/red]\n"
            "Run the ingestion/pipeline first, or pass --graph-file."
        )
        raise typer.Exit(code=1)

    return G  # type: ignore[return-value]


def _find_paper_node(graph: nx.MultiDiGraph, arxiv_id: str) -> Optional[Any]:
    """
    Robustly find the node corresponding to this paper id in different graph flavors:
      - node id == paper_node_id(arxiv_id) (e.g. 'paper:p1')
      - node id == arxiv_id
      - node with node['arxiv_id'] == arxiv_id or node['paper_id'] == arxiv_id
    """
    node = paper_node_id(arxiv_id)
    if node in graph:
        return node

    if arxiv_id in graph:
        return arxiv_id

    for n, attrs in graph.nodes(data=True):
        if attrs.get("arxiv_id") == arxiv_id or attrs.get("paper_id") == arxiv_id:
            return n

    return None


def _node_kind_and_label(graph: nx.MultiDiGraph, node: Any) -> Tuple[str, str]:
    attrs: Dict[str, Any] = graph.nodes[node]
    kind = str(attrs.get("type") or attrs.get("kind") or "").lower() or "unknown"

    label = (
        attrs.get("title")
        or attrs.get("label")
        or attrs.get("name")
        or attrs.get("concept_key")
        or str(node)
    )

    return kind, str(label)


def _iter_concept_nodes(graph: nx.MultiDiGraph) -> List[Tuple[Any, Dict[str, Any]]]:
    concepts: List[Tuple[Any, Dict[str, Any]]] = []
    for n, attrs in graph.nodes(data=True):
        type_str = str(attrs.get("type", "")).lower()
        kind_str = str(attrs.get("kind", "")).lower()
        if type_str == "concept" or "concept" in kind_str:
            concepts.append((n, attrs))
    return concepts


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("search")
def search(
    query: str = typer.Argument(
        ..., help="Search text to match against node ids and labels."
    ),
    kind: Optional[str] = typer.Option(
        None,
        "--kind",
        "-k",
        help="Optional node kind filter (e.g. 'paper', 'concept').",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        "-n",
        min=1,
        help="Max number of hits to display.",
    ),
    graph_file: Optional[Path] = typer.Option(
        None,
        "--graph-file",
        "-g",
        help=(
            "Path to a saved graph file. "
            "If omitted, the latest graph in settings.graph_dir is used."
        ),
    ),
) -> None:
    """
    Search nodes by label or id, optionally filtered by kind.
    """
    G = _load_graph(graph_file)

    q = query.lower()
    kind_filter = kind.lower().strip() if kind else None

    hits: List[Tuple[str, str, str]] = []
    for node, attrs in G.nodes(data=True):
        node_id = str(node)
        node_kind, label = _node_kind_and_label(G, node)

        if kind_filter and node_kind != kind_filter:
            continue

        if q in node_id.lower() or q in label.lower():
            hits.append((node_id, node_kind, label))

        if len(hits) >= limit:
            break

    if not hits:
        console.print(f"[yellow]No matches for '{query}'.[/yellow]")
        return

    console.print(
        f"[bold]Search results for '{query}'"
        f"{f' (kind={kind_filter})' if kind_filter else ''}:[/bold]"
    )

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Node id")
    tbl.add_column("Kind")
    tbl.add_column("Label")

    for node_id, node_kind, label in hits:
        tbl.add_row(node_id, node_kind, label)

    console.print(tbl)


@app.command("paper")
def paper(
    arxiv_id: str = typer.Argument(
        ..., help="arXiv id / paper id (e.g. 2401.12345 or p1)."
    ),
    graph_file: Optional[Path] = typer.Option(
        None,
        "--graph-file",
        "-g",
        help=(
            "Path to a saved graph pickle. "
            "If omitted, the latest graph in settings.graph_dir is used."
        ),
    ),
    top_k_concepts: int = typer.Option(
        10,
        "--top-k-concepts",
        help="Max number of concepts to show.",
    ),
    top_k_references: int = typer.Option(
        10,
        "--top-k-refs",
        help="Max number of influential references to show.",
    ),
    top_k_influenced: int = typer.Option(
        10,
        "--top-k-influenced",
        help="Max number of papers influenced by this one.",
    ),
) -> None:
    """
    Show a compact influence/concept view for a single paper.
    """
    G = _load_graph(graph_file)

    view = get_paper_influence_view(
        G,
        arxiv_id,
        top_k_concepts=top_k_concepts,
        top_k_references=top_k_references,
        top_k_influenced=top_k_influenced,
    )

    if view is None or view.paper is None:
        console.print(f"[red]Paper '{arxiv_id}' not found in graph.[/red]")
        raise typer.Exit(code=1)

    paper_summary = view.paper

    console.print(
        f"[bold]Paper {paper_summary.arxiv_id}[/bold]: "
        f"{paper_summary.title or '(no title)'}"
    )

    abstract = getattr(paper_summary, "abstract", None)
    if abstract:
        console.print(f"[dim]{abstract}[/dim]\n")

    console.print("[bold]Top concepts:[/bold]")
    if not view.concepts:
        console.print("  (none)")
    else:
        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("Concept")
        tbl.add_column("Weight", justify="right")
        tbl.add_column("Details")

        for c in view.concepts:
            weight = getattr(c, "weight", None)
            weight_str = f"{weight:.3f}" if isinstance(weight, (int, float)) else ""
            details = []
            if getattr(c, "mentions_total", None) is not None:
                details.append(f"mentions={c.mentions_total}")
            tbl.add_row(c.label, weight_str, ", ".join(details))

        console.print(tbl)

    console.print("\n[bold]Influential references (papers this one cites):[/bold]")
    refs = getattr(view, "influential_references", None) or []
    if not refs:
        console.print("  (none)")
    else:
        for r in refs:
            title = getattr(r, "title", "")
            title_part = f" — {title}" if title else ""
            console.print(
                f"  • {r.arxiv_id}{title_part} "
                f"(influence_score={r.influence_score:.3f})"
            )

    console.print("\n[bold]Papers influenced by this one:[/bold]")
    influenced = getattr(view, "influenced_papers", None) or []
    if not influenced:
        console.print("  (none)")
    else:
        for r in influenced:
            title = getattr(r, "title", "")
            title_part = f" — {title}" if title else ""
            console.print(
                f"  • {r.arxiv_id}{title_part} "
                f"(influence_score={r.influence_score:.3f})"
            )


@app.command("neighbors")
def neighbors(
    arxiv_id: str = typer.Argument(
        ..., help="arXiv id / paper id whose neighborhood to inspect."
    ),
    depth: int = typer.Option(
        1,
        "--depth",
        "-d",
        min=1,
        help="Graph distance to traverse (1 = direct neighbors).",
    ),
    graph_file: Optional[Path] = typer.Option(
        None,
        "--graph-file",
        "-g",
        help="Path to a saved graph pickle.",
    ),
) -> None:
    """
    Show the neighborhood around a paper up to a given depth.
    """
    G = _load_graph(graph_file)
    start = _find_paper_node(G, arxiv_id)

    if start is None:
        console.print(f"[red]Paper '{arxiv_id}' not found in graph.[/red]")
        raise typer.Exit(code=1)

    visited = {start}
    frontier: List[Tuple[Any, int]] = [(start, 0)]
    results: List[Tuple[Any, int]] = []

    while frontier:
        node, d = frontier.pop(0)
        if d >= depth:
            continue

        neighbors_set = set(G.predecessors(node)) | set(G.successors(node))
        for nbr in neighbors_set:
            if nbr in visited:
                continue
            visited.add(nbr)
            nd = d + 1
            results.append((nbr, nd))
            frontier.append((nbr, nd))

    console.print(
        f"[bold]Neighbors of '{arxiv_id}'[/bold] "
        f"(graph node {start}, depth ≤ {depth}):"
    )

    if not results:
        console.print("  (no neighbors)")
        return

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Node id")
    tbl.add_column("Kind")
    tbl.add_column("Label")
    tbl.add_column("Depth", justify="right")

    for node, d in sorted(results, key=lambda x: x[1]):
        kind, label = _node_kind_and_label(G, node)
        tbl.add_row(str(node), kind, label, str(d))

    console.print(tbl)


@app.command("top-concepts")
def top_concepts(
    limit: int = typer.Option(
        10,
        "--limit",
        "-k",
        min=1,
        help="Number of top concepts to show (by graph degree).",
    ),
    graph_file: Optional[Path] = typer.Option(
        None,
        "--graph-file",
        "-g",
        help="Path to a saved graph pickle.",
    ),
) -> None:
    """
    Show the highest-degree concept nodes in the graph.
    """
    G = _load_graph(graph_file)

    items: List[Tuple[Any, str, int]] = []
    for n, attrs in _iter_concept_nodes(G):
        degree = int(G.degree(n))
        label = (
            attrs.get("label")
            or attrs.get("name")
            or attrs.get("concept_key")
            or str(n)
        )
        items.append((n, str(label), degree))

    if not items:
        console.print("[yellow]No concept nodes found in graph.[/yellow]")
        return

    items.sort(key=lambda x: x[2], reverse=True)
    items = items[:limit]

    console.print(f"[bold]Top {len(items)} concepts by degree:[/bold]")
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Node id")
    tbl.add_column("Label")
    tbl.add_column("Degree", justify="right")

    for node, label, degree in items:
        tbl.add_row(str(node), label, str(degree))

    console.print(tbl)
