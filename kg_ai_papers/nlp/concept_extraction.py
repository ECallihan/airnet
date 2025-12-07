# kg_ai_papers/nlp/concept_extraction.py

from __future__ import annotations

from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple, Optional, Iterable
from kg_ai_papers.config.settings import Settings, RuntimeMode

from kg_ai_papers.tei_parser import PaperSection
from kg_ai_papers.documents import PaperDocument
from dataclasses import dataclass
from keybert import KeyBERT

from kg_ai_papers.config.settings import settings
from kg_ai_papers.models.concept import Concept, ConceptOccurrence
from kg_ai_papers.models.paper import Paper

_settings = Settings()

# Section kinds/titles to prioritize in LIGHT mode.
# This is just a heuristic – you can refine it later.
_LIGHT_MODE_IMPORTANT_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "conclusion",
    "summary",
}


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

@dataclass
class SectionConcept:
    """
    A concept mention tied to a specific paper section.

    Attributes:
        concept: The underlying Concept object (name, kind, etc.).
        paper_id: ID of the paper this concept comes from.
        section_title: Title of the section (e.g. "Introduction", "Methods").
        section_level: Section nesting level (1 = top-level, if available).
    """
    concept: "Concept"
    paper_id: Optional[str] = None
    section_title: Optional[str] = None
    section_level: Optional[int] = None

def extract_concepts_from_sections(
    sections: Sequence[PaperSection],
    paper_id: Optional[str] = None,
) -> List[SectionConcept]:
    """
    Run the existing text-based concept extraction over each PaperSection.

    This preserves your existing heuristics in extract_concepts(text),
    but enriches the results with section metadata so downstream
    influence modeling can weigh concepts differently by section.
    """
    results: List[SectionConcept] = []

    for sec in sections:
        if not sec.text:
            continue

        # Re-use your existing function; do NOT change its behavior.
        concepts = extract_concepts(sec.text)

        for c in concepts:
            results.append(
                SectionConcept(
                    concept=c,
                    paper_id=paper_id,
                    section_title=sec.title or None,
                    section_level=sec.level,
                )
            )

    return results


def extract_concepts_from_document(
    doc: "PaperDocument",
    settings: Optional[Settings] = None,
) -> List["SectionConcept"]:
    """
    Extract concepts from a PaperDocument, with behavior modulated by runtime_mode.

    STANDARD (default):
        - Process all sections (current behavior).

    LIGHT:
        - Focus on the most important sections (abstract, intro, related work, conclusion).
        - If those can’t be identified, fall back to the first few sections.

    HEAVY:
        - For now, same as STANDARD. Later you might:
            * increase candidate limits
            * add extra passes (e.g., formula/concept extraction)
    """
    settings = settings or _settings
    sections: Iterable["PaperSection"] = doc.sections

    if settings.runtime_mode == RuntimeMode.LIGHT:
        # Prefer sections whose kind or title matches the important set
        filtered = [
            s
            for s in sections
            if (
                s.kind
                and s.kind.lower() in _LIGHT_MODE_IMPORTANT_SECTIONS
            ) or (
                s.title
                and s.title.lower() in _LIGHT_MODE_IMPORTANT_SECTIONS
            )
        ]

        # Fallback: if nothing matched, keep just the first few sections
        if filtered:
            sections_to_use = filtered
        else:
            # doc.sections is likely a list; if not, cast to list
            all_sections = list(doc.sections)
            sections_to_use = all_sections[:3]
    else:
        # STANDARD and HEAVY both process all sections for now
        sections_to_use = doc.sections

    concepts: List["SectionConcept"] = []
    for section in sections_to_use:
        concepts.extend(extract_concepts_from_section(section))

    return concepts


@dataclass
class ConceptSummary:
    """
    Aggregated view of a concept within a single paper.

    Attributes:
        concept_key: A stable key for the concept
                     (typically concept.name or (name, kind)).
        base_name: Human-readable name for the concept.
        kind: Optional concept kind/type (e.g. "method", "dataset").
        mentions_total: Total number of mentions across all sections.
        mentions_by_section: Raw counts per section title.
        weighted_score: Section-weighted importance score for the concept.
    """
    concept_key: str
    base_name: str
    kind: Optional[str]
    mentions_total: int
    mentions_by_section: Dict[str, int] = field(default_factory=dict)
    weighted_score: float = 0.0

def _default_section_weights() -> Dict[str, float]:
    """
    Default heuristic weights per section title (case-insensitive).

    You can override these in aggregate_section_concepts if needed.
    """
    # These are normalized by lowercase and simple substring checks.
    return {
        "introduction": 0.5,
        "background": 0.7,
        "related work": 0.7,
        "preliminaries": 0.7,
        "method": 1.5,      # matches "methods", "methodology", etc.
        "approach": 1.5,
        "model": 1.5,
        "experiment": 1.3,
        "results": 1.3,
        "evaluation": 1.3,
        "analysis": 1.2,
        "discussion": 0.8,
        "conclusion": 0.8,
        "future work": 0.8,
    }


def _lookup_section_weight(
    section_title: Optional[str],
    weights: Mapping[str, float],
    default_weight: float = 1.0,
) -> float:
    """
    Map a section title to a weight using fuzzy, case-insensitive substring rules.
    """
    if not section_title:
        return default_weight

    title = section_title.lower()

    # First, try exact normalized keys
    if title in weights:
        return weights[title]

    # Otherwise, use substring matches (e.g., "introduction and overview")
    # We keep it simple and deterministic: first match wins, based on insertion order.
    for key, w in weights.items():
        if key in title:
            return w

    return default_weight


def aggregate_section_concepts(
    section_concepts: Sequence["SectionConcept"],
    section_weights: Optional[Mapping[str, float]] = None,
    default_section_weight: float = 1.0,
) -> Dict[str, ConceptSummary]:
    """
    Aggregate SectionConcept objects into per-concept summaries for a single paper.

    Returns:
        A dict mapping concept_key -> ConceptSummary.

    Notes:
        - concept_key defaults to concept.name if available, otherwise str(concept).
        - If your Concept dataclass has a 'kind' attribute, it is used; otherwise None.
        - weighted_score = sum_over_mentions(section_weight_for_that_section).
    """
    if section_weights is None:
        section_weights = _default_section_weights()

    # concept_key -> (base_name, kind, total_count, dict[section_title] -> count, weighted_score)
    aggregates: Dict[str, Tuple[str, Optional[str], int, Dict[str, int], float]] = {}

    for sc in section_concepts:
        concept = sc.concept

        # Try to be robust to whatever your existing Concept dataclass looks like
        name = getattr(concept, "name", None) or str(concept)
        kind = getattr(concept, "kind", None)

        concept_key = name  # you can later change this to (name, kind) if needed

        section_title = sc.section_title or "UNKNOWN"
        weight = _lookup_section_weight(section_title, section_weights, default_section_weight)

        if concept_key not in aggregates:
            aggregates[concept_key] = (
                name,            # base_name
                kind,            # kind
                0,               # total_count
                defaultdict(int),# mentions_by_section
                0.0,             # weighted_score
            )

        base_name, concept_kind, total, per_section, weighted_score = aggregates[concept_key]

        total += 1
        per_section[section_title] += 1
        weighted_score += weight

        aggregates[concept_key] = (base_name, concept_kind, total, per_section, weighted_score)

    # Convert into ConceptSummary objects
    summaries: Dict[str, ConceptSummary] = {}
    for key, (base_name, concept_kind, total, per_section, weighted_score) in aggregates.items():
        summaries[key] = ConceptSummary(
            concept_key=key,
            base_name=base_name,
            kind=concept_kind,
            mentions_total=total,
            mentions_by_section=dict(per_section),
            weighted_score=weighted_score,
        )

    return summaries


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
