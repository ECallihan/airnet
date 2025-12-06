# kg_ai_papers/graph/storage.py

from __future__ import annotations

from pathlib import Path
import networkx as nx

from kg_ai_papers.config.settings import settings


def save_graph(G: nx.MultiDiGraph, name: str = "graph") -> Path:
    path = settings.graph_dir / f"{name}.gpickle"
    nx.write_gpickle(G, path)
    return path


def load_graph(name: str = "graph") -> nx.MultiDiGraph:
    path = settings.graph_dir / f"{name}.gpickle"
    return nx.read_gpickle(path)
