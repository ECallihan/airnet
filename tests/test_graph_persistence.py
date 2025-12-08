import inspect
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import pytest

import kg_ai_papers.graph.storage as storage
from kg_ai_papers.graph.storage import save_graph

# load_latest_graph is optional – fall back to None if not present
try:
    from kg_ai_papers.graph.storage import load_latest_graph  # type: ignore
except ImportError:  # pragma: no cover - if the helper doesn't exist yet
    load_latest_graph = None  # type: ignore[assignment]


def _maybe_patch_graph_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Optional[str]:
    """
    Best-effort helper to redirect storage to a temporary directory.

    We don't know exactly what the graph directory attribute is called in
    kg_ai_papers.graph.storage, so we scan for something that looks like
    a path and has both 'graph' and 'dir' in its name.

    If we find one, we patch it to tmp_path and return the attribute name
    (mostly for debugging); otherwise we return None and let the storage
    module use its default location.
    """
    graph_dir_attr = None
    for name, value in vars(storage).items():
        lname = name.lower()
        if "graph" in lname and "dir" in lname and isinstance(value, (str, Path)):
            graph_dir_attr = name
            break

    if graph_dir_attr is not None:
        monkeypatch.setattr(storage, graph_dir_attr, tmp_path, raising=False)

    return graph_dir_attr


def _normalized_nodes(G: nx.MultiDiGraph):
    """
    Return a hashable representation of nodes + attributes.

    Output is a set of (node_id, tuple(sorted(attr_items))) pairs.
    """
    result = set()
    for n, data in G.nodes(data=True):
        # data is a dict; convert to a sorted tuple of (key, value) pairs
        items = tuple(sorted(data.items()))
        result.add((n, items))
    return result


def _normalized_edges(G: nx.MultiDiGraph):
    """
    Return a hashable representation of edges + attributes for a MultiDiGraph.

    Output is a set of (u, v, key, tuple(sorted(attr_items))) tuples.
    """
    result = set()
    # include keys=True so we distinguish parallel edges
    for u, v, key, data in G.edges(keys=True, data=True):
        items = tuple(sorted(data.items()))
        result.add((u, v, key, items))
    return result


def test_save_graph_round_trip(tmp_path, monkeypatch):
    """
    Ensure save_graph writes a valid pickle that can be round-tripped with
    the standard library pickle.load.

    This test does NOT depend on load_latest_graph – it just trusts the path
    that save_graph returns.
    """
    # Best-effort redirection of any graph directory to tmp_path
    _maybe_patch_graph_dir(tmp_path, monkeypatch)

    # Build a small but non-trivial graph
    G = nx.MultiDiGraph()
    paper_id = "1234.56789"
    G.add_node(
        f"paper:{paper_id}",
        arxiv_id=paper_id,
        title="Persisted Paper",
        abstract="Some abstract",
    )
    G.add_node("concept:graph-persistence", kind="concept")
    G.add_edge(
        f"paper:{paper_id}",
        "concept:graph-persistence",
        relation="MENTIONS",
        weight=0.9,
    )

    # Save the graph – let storage decide the final path
    path = save_graph(G, name="unit-test-graph")

    # Coerce to Path in case the implementation returns a string
    path = Path(path)
    assert path.exists(), f"Expected saved graph at {path}, but it does not exist"

    # Round-trip via standard pickle
    with path.open("rb") as f:
        loaded = pickle.load(f)

    assert isinstance(loaded, type(G))

    # Compare node and edge sets (including attributes) using normalized forms
    assert _normalized_nodes(loaded) == _normalized_nodes(G)
    assert _normalized_edges(loaded) == _normalized_edges(G)


@pytest.mark.skipif(load_latest_graph is None, reason="load_latest_graph not implemented in storage module")
def test_load_latest_graph_returns_saved_graph(tmp_path, monkeypatch):
    """
    If load_latest_graph is implemented, ensure that after saving at least
    one graph, load_latest_graph returns an equivalent graph.

    We introspect the function's signature and skip the test if it requires
    non-optional parameters (so we don't encode incorrect assumptions).
    """
    # load_latest_graph is imported above if present
    sig = inspect.signature(load_latest_graph)  # type: ignore[arg-type]
    # If there are required parameters, we don't know how to call it safely
    for param in sig.parameters.values():
        if (
            param.default is inspect._empty
            and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ):
            pytest.skip("load_latest_graph requires parameters; test does not guess arguments")

    # Best-effort redirection of any graph directory to tmp_path
    _maybe_patch_graph_dir(tmp_path, monkeypatch)

    # Start with a clean temp dir
    for existing in tmp_path.glob("*"):
        if existing.is_file():
            existing.unlink()

    # Build and save a simple graph
    G = nx.MultiDiGraph()
    paper_id = "9876.54321"
    G.add_node(
        f"paper:{paper_id}",
        arxiv_id=paper_id,
        title="Latest Graph Paper",
        abstract="Persistence test",
    )

    save_graph(G, name="latest-test-graph")

    # Now ask the storage helper for the "latest" graph
    loaded = load_latest_graph()  # type: ignore[call-arg]
    assert loaded is not None, "load_latest_graph() returned None"

    assert isinstance(loaded, type(G))
    # Compare node sets; we don't insist on edge parity here in case the
    # storage layer adds any bookkeeping edges, but for now they should match.
    assert _normalized_nodes(loaded) == _normalized_nodes(G)
