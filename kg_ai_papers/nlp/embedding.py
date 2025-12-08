"""
Embedding utilities for papers.

We wrap SentenceTransformer in a small EmbeddingModel helper so that:
- Tests can call `get_embedding_model().encode_paper(paper)`.
- Configuration (model name, device, batch size) comes from Settings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence

import torch
from sentence_transformers import SentenceTransformer

from kg_ai_papers.config.settings import get_settings
from kg_ai_papers.models.paper import Paper


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer that knows how to
    embed papers and plain text.
    """

    def __init__(self, backend: SentenceTransformer, batch_size: int) -> None:
        self._backend = backend
        self._batch_size = batch_size

    # ---- Low-level helpers -------------------------------------------------

    def encode_texts(self, texts: Sequence[str]):
        """
        Encode a batch of texts into embeddings.

        Returns whatever the underlying SentenceTransformer returns
        (in practice: a tensor or numpy array, depending on settings).
        """
        if not texts:
            return []

        return self._backend.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def encode_text(self, text: str):
        """
        Encode a single text string.
        """
        return self.encode_texts([text])[0]

    # ---- Paper-aware helper ------------------------------------------------

    def encode_paper(self, paper: Paper):
        """
        Encode a Paper into a single embedding.

        We concatenate the title + abstract so that tests like
        test_embedding_similarity can call:

            embedder = get_embedding_model()
            p1.embedding = embedder.encode_paper(p1)
        """
        parts: List[str] = []
        if paper.title:
            parts.append(paper.title)
        if paper.abstract:
            parts.append(paper.abstract)

        text = "\n\n".join(parts).strip()
        # Safeguard: still handle completely empty papers
        if not text:
            text = ""

        return self.encode_text(text)


# ---- Backend construction ---------------------------------------------------


def _resolve_device(raw_device: str) -> str:
    """
    Turn the EMBEDDING_DEVICE setting into an actual device string.

    - "auto"  -> "cuda" if available else "cpu"
    - "cpu"   -> "cpu"
    - "cuda"  -> "cuda"
    - anything else is passed through as-is.
    """
    if raw_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw_device


@lru_cache(maxsize=1)
def _build_backend() -> EmbeddingModel:
    """
    Build and cache the underlying SentenceTransformer-based model,
    configured from Settings.
    """
    settings = get_settings()

    model_name = settings.SENTENCE_MODEL_NAME
    device = _resolve_device(settings.EMBEDDING_DEVICE)
    batch_size = settings.EMBEDDING_BATCH_SIZE

    backend = SentenceTransformer(model_name, device=device)
    return EmbeddingModel(backend=backend, batch_size=batch_size)


def get_embedding_model() -> EmbeddingModel:
    """
    Public entry point used throughout the codebase and in tests.

    Returns an EmbeddingModel instance with an `encode_paper` method.
    """
    return _build_backend()


# ---- Convenience functions (kept for callers that used the old API) ---------


def embed_texts(texts: Iterable[str]):
    """
    Convenience wrapper: embed a sequence of texts using the global model.
    """
    model = get_embedding_model()
    return model.encode_texts(list(texts))


def embed_text(text: str):
    """
    Convenience wrapper: embed a single text string.
    """
    model = get_embedding_model()
    return model.encode_text(text)
