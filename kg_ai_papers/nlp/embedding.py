# kg_ai_papers/nlp/embedding.py

from __future__ import annotations

from typing import List, Iterable, Optional
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None

from kg_ai_papers.config.settings import settings
from kg_ai_papers.models.paper import Paper
from kg_ai_papers.models.section import Section


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer for:
      - text embeddings
      - paper-level embeddings (title + abstract + sections)
      - concept label embeddings

    Keeps a single underlying model instance (per process),
    and lets you choose device (CPU / CUDA) if available.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        :param model_name: SentenceTransformer model name.
        :param device: "cpu", "cuda", or None to auto-detect.
        """
        if device is None:
            device = self._auto_device()
        self.device = device

        self._model = SentenceTransformer(model_name, device=self.device)

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_device() -> str:
        """
        Simple heuristic:
          - If torch with CUDA is available, use "cuda"
          - Else fallback to "cpu"
        """
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ------------------------------------------------------------------
    # Core encoding helpers
    # ------------------------------------------------------------------

    def encode_text(
        self,
        text: str,
        normalize: bool = True,
        convert_to_tensor: bool = True,
    ):
        """
        Encode a single text string.
        Returns a torch.Tensor by default (good for cosine similarity).
        """
        if not text:
            text = ""

        emb = self._model.encode(
            text,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize,
        )
        return emb

    def encode_texts(
        self,
        texts: Iterable[str],
        normalize: bool = True,
        convert_to_tensor: bool = True,
        batch_size: int = 32,
    ):
        """
        Encode a batch of texts more efficiently than looping.
        Returns either a torch.Tensor or np.ndarray depending on convert_to_tensor.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        embs = self._model.encode(
            texts_list,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embs

    # ------------------------------------------------------------------
    # Concept embeddings
    # ------------------------------------------------------------------

    def encode_concept_label(
        self,
        label: str,
        normalize: bool = True,
        convert_to_tensor: bool = True,
    ):
        """
        Encode a concept label (e.g., "graph neural networks").
        """
        return self.encode_text(
            label,
            normalize=normalize,
            convert_to_tensor=convert_to_tensor,
        )

    # ------------------------------------------------------------------
    # Paper-level embeddings
    # ------------------------------------------------------------------

    def _build_paper_text(
        self,
        paper: Paper,
        max_chars: int = 8000,
        include_sections: bool = True,
        max_sections: int = 5,
    ) -> str:
        """
        Build a representative text for a paper:
          - title
          - abstract
          - first few sections (if available)
        Truncate to max_chars to avoid huge inputs.
        """
        parts: List[str] = []

        if paper.title:
            parts.append(paper.title)

        if paper.abstract:
            parts.append(paper.abstract)

        if include_sections and paper.sections:
            # Take first N sections (often Intro, Background, etc.)
            for section in paper.sections[:max_sections]:
                if section.title:
                    parts.append(section.title)
                if section.text:
                    parts.append(section.text)

        full_text = "\n\n".join(parts)
        return full_text[:max_chars]

    def encode_paper(
        self,
        paper: Paper,
        normalize: bool = True,
        convert_to_tensor: bool = True,
        include_sections: bool = True,
        max_sections: int = 5,
        max_chars: int = 8000,
    ):
        """
        Encode a paper as a single embedding.
        """
        text = self._build_paper_text(
            paper=paper,
            max_chars=max_chars,
            include_sections=include_sections,
            max_sections=max_sections,
        )
        emb = self.encode_text(
            text,
            normalize=normalize,
            convert_to_tensor=convert_to_tensor,
        )
        return emb

    def encode_papers_batch(
        self,
        papers: List[Paper],
        normalize: bool = True,
        convert_to_tensor: bool = True,
        include_sections: bool = True,
        max_sections: int = 5,
        max_chars: int = 8000,
        batch_size: int = 16,
    ):
        """
        Efficiently encode a batch of papers.
        Returns an array/tensor of embeddings and also sets paper.embedding.
        """
        if not papers:
            return []

        texts = [
            self._build_paper_text(
                paper=p,
                max_chars=max_chars,
                include_sections=include_sections,
                max_sections=max_sections,
            )
            for p in papers
        ]

        embs = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        # Attach embeddings back to papers
        if convert_to_tensor:
            # embs is usually a single 2D torch.Tensor
            for i, p in enumerate(papers):
                p.embedding = embs[i]
        else:
            # embs is a np.ndarray
            for i, p in enumerate(papers):
                p.embedding = embs[i, :]

        return embs


# ----------------------------------------------------------------------
# Singleton accessor
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """
    Cached EmbeddingModel instance using the model name from settings.
    """
    return EmbeddingModel(model_name=settings.SENTENCE_MODEL_NAME)
