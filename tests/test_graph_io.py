import networkx as nx

from kg_ai_papers.graph.io import load_graph, save_graph


def test_save_and_load_roundtrip(tmp_path):
    # Arrange: build a small MultiDiGraph with attributes
    G = nx.MultiDiGraph()
    paper_node = "paper:2401.01234"
    concept_node = "concept:test"

    G.add_node(paper_node, kind="paper", arxiv_id="2401.01234")
    G.add_node(concept_node, kind="concept", label="test concept")
    G.add_edge(
        paper_node,
        concept_node,
        key="HAS_CONCEPT",
        relation="HAS_CONCEPT",
        weight=0.9,
    )

    # Act: save without explicit suffix; should default to .gpickle
    out_path = tmp_path / "airnet_graph"
    saved_path = save_graph(G, out_path)

    # Assert: file exists and has .gpickle extension
    assert saved_path.exists()
    assert saved_path.suffix == ".gpickle"

    # Act 2: load it back
    G_loaded = load_graph(saved_path)

    # Assert structural equivalence
    assert isinstance(G_loaded, nx.MultiDiGraph)
    assert set(G_loaded.nodes) == set(G.nodes)
    assert set(G_loaded.edges(keys=True)) == set(G.edges(keys=True))

    # Assert attributes survived the roundtrip
    assert G_loaded.nodes[paper_node]["kind"] == "paper"
    assert G_loaded.nodes[paper_node]["arxiv_id"] == "2401.01234"
    assert G_loaded.nodes[concept_node]["label"] == "test concept"

    # Check one edge attribute
    (u, v, k, data_loaded) = next(iter(G_loaded.edges(keys=True, data=True)))
    assert data_loaded["relation"] == "HAS_CONCEPT"
    assert data_loaded["weight"] == 0.9


def test_load_graph_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.gpickle"
    try:
        load_graph(missing)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised, "Expected FileNotFoundError for missing graph file"


def test_save_graph_no_overwrite(tmp_path):
    G = nx.Graph()
    G.add_node("x")

    path = tmp_path / "graph.gpickle"
    save_graph(G, path)

    # Second call with overwrite=False should raise
    raised = False
    try:
        save_graph(G, path, overwrite=False)
    except FileExistsError:
        raised = True

    assert raised, "Expected FileExistsError when overwrite=False and file exists"
