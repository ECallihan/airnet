# kg_ai_papers/graph/storage.py

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from kg_ai_papers.config.settings import settings


def save_graph(G: nx.MultiDiGraph, name: str = None) -> Path:
    """
    Save the graph to disk using Python pickle.

    Args:
        G: The graph to save.
        name: Logical graph name (without extension). Defaults to settings.GRAPH_DEFAULT_NAME.
    """
    name = name or settings.GRAPH_DEFAULT_NAME
    path = settings.graph_dir / f"{name}.gpickle"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(G, f)

    return path


def load_graph(name: str = None) -> nx.MultiDiGraph:
    """
    Load the graph from disk using Python pickle.

    Args:
        name: Logical graph name (without extension). Defaults to settings.GRAPH_DEFAULT_NAME.

    Raises:
        FileNotFoundError if the graph file does not exist.
    """
    name = name or settings.GRAPH_DEFAULT_NAME
    path = settings.graph_dir / f"{name}.gpickle"

    with path.open("rb") as f:
        G: nx.MultiDiGraph = pickle.load(f)

    return G
