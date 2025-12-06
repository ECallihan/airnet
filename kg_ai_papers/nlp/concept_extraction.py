# kg_ai_papers/nlp/concept_extraction.py

from __future__ import annotations

from typing import List, Iterable, Tuple, Dict
from functools import lru_cache

from keybert import KeyBERT

from kg_ai_papers.config.settings import settings
from kg_ai_papers.models.paper import Paper
from kg_ai_papers.models.section import Section
from kg_ai_papers.models.concept import ConceptOccurrence


class ConceptExtractor:
    """
    Thin wrapper around KeyBERT (and optionally other methods) to extract
    keyphrases as 'concepts' at the section and paper level.
    """

    def __init__(self, model_name: str | None = None):
        """
        :param model_name: underlying embedding model to use for KeyBERT.
                           If None, KeyBERT uses its default model.
        """
        if model_name:
            self.kw_model = KeyBERT(model=model_name)
        else:
            self.kw_model = KeyBERT()

    # ------------------------
    # Low-level helper
    # ------------------------

    def _extract_from_text(
        self,
        text: str,
        top_n: int,
        keyphrase_ngram_range: Tuple[int, int] = (1, 3),
    ) -> List[Tuple[str, float]]:
        if not text or not text.strip():
            return []
        # Truncate extremely long text for speed
        text = text[: settings.MAX_SECTION_CHARS]
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words="english",
            top_n=top_n,
        )
        # keywords is List[(phrase, score)]
        return [(phrase.strip(), float(score)) for phrase, score in keywords if phrase.strip()]

    # ------------------------
    # Public API
    # ------------------------

    def extract_for_section(
        self,
        paper_id: str,
        section: Section,
        top_n: int | None = None,
    ) -> List[ConceptOccurrence]:
        """
        Extract concepts from a single section.
        """
        if top_n is None:
            top_n = settings.SECTION_TOP_CONCEPTS

        raw_concepts = self._extract_from_text(section.text, top_n=top_n)
        occurrences: List[ConceptOccurrence] = []
        for idx, (label, score) in enumerate(raw_concepts):
            occurrences.append(
                ConceptOccurrence(
                    paper_id=paper_id,
                    label=label,
                    score=score,
                    source_section_title=section.title,
                    source_section_level=section.level,
                    position_in_section=idx,
                )
            )
        return occurrences

    def extract_for_sections(
        self,
        paper_id: str,
        sections: Iterable[Section],
        section_top_n: int | None = None,
    ) -> List[ConceptOccurrence]:
        """
        Extract concepts for each section in a paper.
        """
        all_occurrences: List[ConceptOccurrence] = []
        for section in sections:
            occs = self.extract_for_section(
                paper_id=paper_id,
                section=section,
                top_n=section_top_n,
            )
            all_occurrences.extend(occs)
        return all_occurrences

    def aggregate_paper_level_concepts(
        self,
        occurrences: Iterable[ConceptOccurrence],
        paper_top_n: int | None = None,
    ) -> List[Tuple[str, float]]:
        """
        Aggregate section-level concept occurrences into a paper-level
        concept ranking.

        Simple heuristic:
          - Sum scores over all occurrences of the same label
          - Optionally, boost concepts that appear in early sections (intro)
        """
        if paper_top_n is None:
            paper_top_n = settings.PAPER_TOP_CONCEPTS

        agg: Dict[str, float] = {}

        for occ in occurrences:
            label = occ.label
            score = occ.score

            # Optional: simple positional boost
            # If the concept appears in early sections, bump its weight slightly
            if occ.source_section_level and occ.source_section_level in {"1", "01"}:
                score *= 1.1

            # You could also weight by (1 / (position_in_section + 1)) if you like.

            agg[label] = agg.get(label, 0.0) + score

        # Sort and take top_n
        sorted_items = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:paper_top_n]

    def extract_for_paper(
        self,
        paper: Paper,
        section_top_n: int | None = None,
        paper_top_n: int | None = None,
    ) -> Paper:
        """
        Full pipeline for one Paper:
          - Use its sections to get section-level ConceptOccurrences
          - Aggregate to paper-level concept list
          - Attach to Paper object
        """
        if not paper.sections:
            # Fallback: use abstract or some text
            # Use a pseudo-section with title "ABSTRACT"
            pseudo_section = Section(
                title="ABSTRACT",
                level="0",
                text=paper.abstract or "",
            )
            occurrences = self.extract_for_section(
                paper_id=paper.arxiv_id,
                section=pseudo_section,
                top_n=section_top_n,
            )
        else:
            occurrences = self.extract_for_sections(
                paper_id=paper.arxiv_id,
                sections=paper.sections,
                section_top_n=section_top_n,
            )

        paper.concepts = occurrences
        # You can also store aggregated paper-level concepts on the Paper, e.g.:
        paper.paper_level_concepts = self.aggregate_paper_level_concepts(
            occurrences,
            paper_top_n=paper_top_n,
        )
        return paper


# ------------------------
# Convenient singleton-style accessor
# ------------------------

@lru_cache(maxsize=1)
def get_concept_extractor() -> ConceptExtractor:
    return ConceptExtractor(model_name=settings.KEYBERT_MODEL_NAME)
