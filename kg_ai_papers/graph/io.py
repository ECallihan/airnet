# kg_ai_papers/graph/io.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import networkx as nx

PathLike = Union[str, Path]


def save_graph(
    graph: nx.Graph,
    path: PathLike,
    overwrite: bool = True,
) -> Path:
    """
    Serialize a NetworkX graph to disk using pickle.

    - If `path` has no suffix, `.gpickle` is appended.
    - Creates parent directories if needed.
    - If `overwrite` is False and the file already exists, raises FileExistsError.
    """
    output_path = Path(path)

    # If no suffix at all (e.g. "airnet_graph"), default to ".gpickle"
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".gpickle")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Respect overwrite flag
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Graph file already exists and overwrite=False: {output_path}")

    with output_path.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path


def load_graph(path: PathLike) -> nx.Graph:
    """
    Load a NetworkX graph from a pickle file.
    """
    p = Path(path)
    with p.open("rb") as f:
        graph = pickle.load(f)
    return graph
