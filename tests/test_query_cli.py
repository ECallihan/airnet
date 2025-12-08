# tests/test_query_cli.py

from pathlib import Path
import pickle

import networkx as nx
from typer.testing import CliRunner

from kg_ai_papers.cli.main import app as cli_app
from kg_ai_papers.graph.schema import NodeType, EdgeType

runner = CliRunner()


def build_toy_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # Paper nodes
    G.add_node(
        "paper:p1",
        type=NodeType.PAPER.value,
        arxiv_id="p1",
        title="Paper One",
        abstract="Abstract one",
    )
    G.add_node(
        "paper:p2",
        type=NodeType.PAPER.value,
        arxiv_id="p2",
        title="Paper Two",
        abstract="Abstract two",
    )

    # Concept node
    G.add_node(
        "concept:c1",
        type=NodeType.CONCEPT.value,
        label="graph neural networks",
    )

    # PAPER_HAS_CONCEPT edge
    G.add_edge(
        "paper:p1",
        "concept:c1",
        type=EdgeType.PAPER_HAS_CONCEPT.value,
        weight=0.9,
    )

    # Citation edge p1 -> p2
    G.add_edge(
        "paper:p1",
        "paper:p2",
        type=EdgeType.PAPER_CITES_PAPER.value,
        influence_score=0.8,
        similarity=0.5,
    )

    return G


def _write_graph(tmp_path: Path) -> Path:
    G = build_toy_graph()
    graph_path = tmp_path / "toy_graph.pkl"
    with graph_path.open("wb") as f:
        pickle.dump(G, f)
    return graph_path


def test_cli_paper_basic(tmp_path):
    graph_path = _write_graph(tmp_path)

    result = runner.invoke(
        cli_app,
        ["query", "paper", "p1", "--graph-file", str(graph_path)],
    )

    assert result.exit_code == 0
    out = result.stdout

    # Paper summary
    assert "p1" in out
    assert "Paper One" in out

    # Concepts and references are surfaced
    assert "graph neural networks" in out
    assert "p2" in out  # referenced paper


def test_cli_neighbors(tmp_path):
    graph_path = _write_graph(tmp_path)

    result = runner.invoke(
        cli_app,
        ["query", "neighbors", "p1", "--graph-file", str(graph_path), "--depth", "1"],
    )

    assert result.exit_code == 0
    out = result.stdout

    # Direct neighbors: concept and cited paper
    assert "paper:p2" in out
    assert "concept:c1" in out


def test_cli_top_concepts(tmp_path):
    graph_path = _write_graph(tmp_path)

    result = runner.invoke(
        cli_app,
        ["query", "top-concepts", "--graph-file", str(graph_path), "--limit", "5"],
    )

    assert result.exit_code == 0
    out = result.stdout

    assert "concept:c1" in out
    assert "graph neural networks" in out
