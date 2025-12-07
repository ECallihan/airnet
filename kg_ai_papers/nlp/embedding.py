from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Optional

import numpy as np

from kg_ai_papers.config.settings import RuntimeMode, get_settings

try:
    # sentence-transformers is expected in normal / full runs
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - only hit in very minimal installs
    SentenceTransformer = None  # type: ignore[misc]


class DummyEmbeddingModel:
    """
    Cheap fallback embedding model used when enable_embeddings=False.

    It produces deterministic pseudo-embeddings based on Python's hash(),
    just to keep shapes and basic similarity behavior consistent for
    downstream code. This is *not* semantically meaningful, but it lets
    the rest of the pipeline run on extremely constrained machines.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(
        self,
        sentences: Iterable[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        **_: object,
    ):
        vecs: List[np.ndarray] = []
        for text in sentences:
            h = hash(text)
            rng = np.random.default_rng(h & 0xFFFFFFFF)
            v = rng.standard_normal(self.dim).astype("float32")
            vecs.append(v)

        arr = np.stack(vecs, axis=0)
        if convert_to_numpy:
            return arr
        return arr  # simple; callers in our codebase expect numpy anyway


@lru_cache(maxsize=1)
def _get_settings():
    # Cached so that multiple embedding calls don’t rebuild Settings.
    return get_settings()


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Returns a singleton embedding model instance.

    Behavior:
    - If settings.enable_embeddings is False -> DummyEmbeddingModel
    - Else -> SentenceTransformer model using
        - settings.embedding_model_name (if set) or
        - settings.SENTENCE_MODEL_NAME
    """
    settings = _get_settings()

    # Lightest possible mode: no real embeddings, just deterministic noise
    if not settings.enable_embeddings:
        return DummyEmbeddingModel()

    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required when enable_embeddings=True. "
            "Either install it or set AIRNET_ENABLE_EMBEDDINGS=false."
        )

    model_name = settings.embedding_model_name or settings.SENTENCE_MODEL_NAME

    # NOTE: we let SentenceTransformer choose device by default.
    # If you later want to do more device-aware logic, we can
    # also inspect settings.runtime_mode and settings.EMBEDDING_DEVICE here.
    model = SentenceTransformer(model_name)
    return model


def embed_texts(texts: Iterable[str]):
    """
    Convenience wrapper that encodes a list/iterable of texts into
    a 2D numpy array of embeddings.

    This is where runtime_mode actually kicks in via
    settings.embedding_batch_size, which selects a batch size based
    on LIGHT / STANDARD / HEAVY.
    """
    settings = _get_settings()
    model = get_embedding_model()

    batch_size = settings.embedding_batch_size
    # For reference:
    # - LIGHT   -> embedding_batch_size_light (e.g. 8)
    # - STANDARD-> embedding_batch_size_standard (e.g. 32)
    # - HEAVY   -> embedding_batch_size_heavy (e.g. 128)

    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
    )
