# kg_ai_papers/nlp/concept_extraction.py

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from keybert import KeyBERT

from kg_ai_papers.config.settings import settings
from kg_ai_papers.models.concept import Concept, ConceptOccurrence
from kg_ai_papers.models.paper import Paper


class ConceptExtractor:
    """
    Thin wrapper around KeyBERT to extract weighted concepts.

    Responsibilities:
    - `extract_concepts(text, top_n)`  -> List[Concept]
    - `extract_for_paper(paper, ...)`  -> mutate Paper with paper- and section-level concepts
    - `enrich_paper(paper)`            -> convenience wrapper used by the pipeline
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        default_top_n: Optional[int] = None,
    ) -> None:
        """
        model_name:
            - If None, use KeyBERT's default underlying model.
            - If a string, pass it through to KeyBERT(model=...).
        """
        if model_name is None:
            # KeyBERT default model
            self._kw_model = KeyBERT()
        else:
            self._kw_model = KeyBERT(model=model_name)

        # default_top_n is only used when callers don't provide explicit top_n
        self.default_top_n = default_top_n or settings.PAPER_TOP_CONCEPTS

    # ------------------------------------------------------------------
    # Core text -> concepts API
    # ------------------------------------------------------------------

    def extract_concepts(
        self,
        text: str,
        top_n: Optional[int] = None,
    ) -> List[Concept]:
        """
        Extract weighted concepts from a block of text.

        Returns a list of `Concept(label, weight)` objects.
        """
        text = (text or "").strip()
        if not text:
            return []

        k = top_n or self.default_top_n

        # KeyBERT returns List[Tuple[str, float]]
        keywords = self._kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_mmr=True,
            diversity=0.5,
            top_n=k,
        )

        concepts: List[Concept] = []
        for phrase, score in keywords:
            phrase = (phrase or "").strip()
            if not phrase:
                continue
            concepts.append(
                Concept(
                    label=phrase,
                    weight=float(score),
                )
            )

        return concepts

    # ------------------------------------------------------------------
    # High-level paper API used by tests + pipeline
    # ------------------------------------------------------------------

    def extract_for_paper(
        self,
        paper: Paper,
        section_top_n: Optional[int] = None,
        paper_top_n: Optional[int] = None,
    ) -> None:
        """
        Mutate `paper` in-place with:
          - paper.paper_level_concepts as List[Tuple[label, score]]
          - paper.concepts (can be a richer list of Concept objects)
          - section-level concepts attached to each Section (as ConceptOccurrence[])
        """
        # ---- paper-level concepts ----
        parts: List[str] = []
        if paper.title:
            parts.append(paper.title)
        if paper.abstract:
            parts.append(paper.abstract)

        paper_text = "\n\n".join(parts).strip()
        if paper_text:
            pl_top = paper_top_n or settings.PAPER_TOP_CONCEPTS
            paper_concept_objs = self.extract_concepts(paper_text, top_n=pl_top)
        else:
            paper_concept_objs = []

        # Expose as list of (label, score) tuples to satisfy tests and
        # tuple-oriented code paths
        paper.paper_level_concepts = [
            (c.label, c.weight) for c in paper_concept_objs
        ]

        # Keep a richer representation in `paper.concepts` if needed
        paper.concepts = paper_concept_objs

        # ---- section-level concepts (if sections exist) ----
        if getattr(paper, "sections", None):
            sec_top = section_top_n or settings.SECTION_TOP_CONCEPTS
            for idx, sec in enumerate(paper.sections):
                title = getattr(sec, "title", "") or ""
                text = getattr(sec, "text", "") or ""
                combined = (title + "\n\n" + text).strip()
                if not combined:
                    continue

                truncated = combined[: settings.MAX_SECTION_CHARS]

                raw_concepts = self.extract_concepts(truncated, top_n=sec_top)

                occurrences: List[ConceptOccurrence] = []
                for c in raw_concepts:
                    occurrences.append(
                        ConceptOccurrence(
                            label=c.label,
                            weight=c.weight,
                            source="section-level",
                            section_id=getattr(sec, "section_id", None) or f"{idx}",
                            paper_id=paper.arxiv_id,
                        )
                    )

                if hasattr(sec, "concepts"):
                    setattr(sec, "concepts", occurrences)
                else:
                    try:
                        sec.concepts = occurrences  # type: ignore[attr-defined]
                    except Exception:
                        pass


    def enrich_paper(self, paper: Paper) -> None:
        """
        Compatibility method used by the parsing pipeline.
        Uses settings for top_n values.
        """
        self.extract_for_paper(
            paper,
            section_top_n=settings.SECTION_TOP_CONCEPTS,
            paper_top_n=settings.PAPER_TOP_CONCEPTS,
        )


# ----------------------------------------------------------------------
# Singleton accessor used by the rest of the codebase
# ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_concept_extractor() -> ConceptExtractor:
    """
    Return a singleton ConceptExtractor.

    Uses KEYBERT_MODEL_NAME from settings by default.
    """
    return ConceptExtractor(
        model_name=settings.KEYBERT_MODEL_NAME,
        default_top_n=settings.PAPER_TOP_CONCEPTS,
    )
