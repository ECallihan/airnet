"""
Utilities for persisting and reloading the in-memory knowledge graph.

This version uses the Python standard library `pickle` module instead of
NetworkX's gpickle helpers, so it works across NetworkX versions that don't
export `write_gpickle` / `read_gpickle` at the top level.

The tests in `tests/test_graph_persistence.py` expect:

- `save_graph(G, name=None, directory=None) -> Path`
    * Writes the graph to disk using pickle.
    * Returns the actual path written.
    * Also maintains a "latest" copy for convenience.

- (Optionally) `load_latest_graph(directory=None) -> Optional[nx.MultiDiGraph]`
    * If present and takes no required parameters, the tests will call it and
      expect it to return the most recently saved graph in that directory.
"""

from __future__ import annotations

import pickle
import shutil
import time
from pathlib import Path
from typing import Optional

import networkx as nx

# Default location to store graph snapshots. The tests will monkeypatch any
# attribute name in this module that looks like a graph directory (contains
# both "graph" and "dir" in its name), so the exact path is not critical.
GRAPH_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "graphs"


def _ensure_dir(directory: Optional[Path]) -> Path:
    """
    Ensure the target directory exists.

    If `directory` is None, fall back to the module-level GRAPH_DIR.
    """
    if directory is None:
        directory = GRAPH_DIR

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_graph(
    G: nx.MultiDiGraph,
    name: Optional[str] = None,
    directory: Optional[Path] = None,
) -> Path:
    """
    Persist the given graph to disk using Python's `pickle` module.

    - If `name` is None, use a timestamp-based filename.
    - Also update a convenient "graph-latest.pkl" copy in the same directory.
    - Returns the full Path to the saved file.

    NOTE: The tests monkeypatch the symbol `save_graph` in `kg_ai_papers.web.app`
    with the signature (G_arg, name=None). This implementation keeps that
    compatible (extra keyword is optional and ignored by tests).
    """
    directory = _ensure_dir(directory)

    if name is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        name = f"graph-{ts}.pkl"

    path = directory / name

    # Serialize with pickle
    with path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Maintain a "latest" convenience copy
    latest_path = directory / "graph-latest.pkl"
    try:
        shutil.copy2(path, latest_path)
    except OSError:
        # If the copy fails for some reason, we still consider the primary
        # save successful.
        pass

    return path


def load_latest_graph(directory: Optional[Path] = None) -> Optional[nx.MultiDiGraph]:
    """
    Load the most recently modified pickled graph file from the target directory.

    - If no files are present, returns None.
    - Uses plain pickle.load for deserialization.
    - This is used by tests if present and if it has no required parameters.
    """
    directory = _ensure_dir(directory)

    # Collect all regular files in the directory
    candidates = [p for p in directory.iterdir() if p.is_file()]
    if not candidates:
        return None

    # Pick the most recently modified file
    latest = max(candidates, key=lambda p: p.stat().st_mtime)

    with latest.open("rb") as f:
        G = pickle.load(f)

    # Type hint for callers
    if not isinstance(G, nx.MultiDiGraph):
        # We don't hard-fail here; callers/tests can assert the type if needed.
        return G  # type: ignore[return-value]

    return G
