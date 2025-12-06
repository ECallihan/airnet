# kg_ai_papers/graph/storage.py

from __future__ import annotations

import pickle
from pathlib import Path
import networkx as nx

from kg_ai_papers.config.settings import settings


def save_graph(G: nx.MultiDiGraph, name: str = None) -> Path:
    """
    Save the graph to disk using Python pickle.
    """
    name = name or settings.GRAPH_DEFAULT_NAME
    path = settings.graph_dir / f"{name}.gpickle"

    with open(path, "wb") as f:
        pickle.dump(G, f)

    return path


def load_graph(name: str = None) -> nx.MultiDiGraph:
    """
    Load the graph from disk using Python pickle.
    """
    name = name or settings.GRAPH_DEFAULT_NAME
    path = settings.graph_dir / f"{name}.gpickle"

    with open(path, "rb") as f:
        G = pickle.load(f)

    return G
