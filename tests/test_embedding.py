# tests/test_embedding.py

from kg_ai_papers.models.paper import Paper
from kg_ai_papers.nlp.embedding import get_embedding_model
from sentence_transformers import util


def test_embedding_similarity():
    p1 = Paper(
        arxiv_id="p1",
        title="Graph Neural Networks",
        abstract="We study graph neural network architectures.",
    )
    p2 = Paper(
        arxiv_id="p2",
        title="Deep Learning on Graphs",
        abstract="We analyze neural networks designed for graph-structured data.",
    )
    p3 = Paper(
        arxiv_id="p3",
        title="Reinforcement Learning for Games",
        abstract="We apply reinforcement learning to play board games.",
    )

    embedder = get_embedding_model()

    p1.embedding = embedder.encode_paper(p1)
    p2.embedding = embedder.encode_paper(p2)
    p3.embedding = embedder.encode_paper(p3)

    sim_12 = util.cos_sim(p1.embedding, p2.embedding).item()
    sim_13 = util.cos_sim(p1.embedding, p3.embedding).item()

    # Graph-related papers should be more similar than graph vs RL-games
    assert sim_12 > sim_13
