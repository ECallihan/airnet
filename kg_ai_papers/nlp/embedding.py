# kg_ai_papers/nlp/embedding.py

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from kg_ai_papers.config.settings import settings
from kg_ai_papers.models.paper import Paper


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer that exposes higher-level helpers
    like encode_paper and similarity, as expected by the tests.
    """

    def __init__(self, model: SentenceTransformer):
        self._model = model

    # --------- low-level helpers ---------

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self._model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_embeddings.extend(emb.tolist())

        return all_embeddings

    def encode_text(self, text: str) -> List[float]:
        return self.encode_texts([text])[0]

    # --------- paper-level helper ---------

    def encode_paper(self, paper: Paper) -> List[float]:
        """
        Encode a Paper using its title + abstract.
        """
        parts = [paper.title or ""]
        if paper.abstract:
            parts.append(paper.abstract)
        text = "\n\n".join(parts).strip()
        if not text:
            # Degenerate case: no text; return a zero vector of some reasonable size
            # We can just encode an empty string to get the right dimensionality.
            text = ""
        return self.encode_text(text)

    # --------- similarity helper ---------

    def similarity(self, v1, v2) -> float:
        """
        Cosine similarity between two vectors (lists, numpy arrays, or tensors).
        """
        v1_arr = np.asarray(v1, dtype=float)
        v2_arr = np.asarray(v2, dtype=float)
        denom = (np.linalg.norm(v1_arr) * np.linalg.norm(v2_arr)) + 1e-8
        if denom == 0.0:
            return 0.0
        return float(np.dot(v1_arr, v2_arr) / denom)


_backend: Optional[EmbeddingModel] = None


def _make_sentence_model() -> SentenceTransformer:
    """
    Construct the underlying SentenceTransformer on the appropriate device.
    """
    if settings.EMBEDDING_DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.EMBEDDING_DEVICE

    return SentenceTransformer(settings.SENTENCE_MODEL_NAME, device=device)


def get_embedding_model() -> EmbeddingModel:
    """
    Return a singleton EmbeddingModel, as expected by tests.
    """
    global _backend
    if _backend is not None:
        return _backend

    st_model = _make_sentence_model()
    _backend = EmbeddingModel(st_model)
    return _backend


# ---------------------------------------------------------------------------
# Convenience functions used by the pipeline
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts and return a list of embedding vectors (as Python lists).
    """
    backend = get_embedding_model()
    return backend.encode_texts(texts)


def embed_text(text: str) -> List[float]:
    """
    Convenience wrapper for a single text.
    """
    backend = get_embedding_model()
    return backend.encode_text(text)
