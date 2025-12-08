# kg_ai_papers/nlp/concept_extraction.py

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Mapping, Sequence, Tuple, Optional, Iterable, Any

from keybert import KeyBERT

from kg_ai_papers.config.settings import settings, Settings, RuntimeMode
from kg_ai_papers.tei_parser import PaperSection
from kg_ai_papers.documents import PaperDocument
from kg_ai_papers.models.concept import Concept, ConceptOccurrence
from kg_ai_papers.models.paper import Paper


# ---------------------------------------------------------------------------
# Runtime-mode helpers
# ---------------------------------------------------------------------------

# Section kinds/titles to prioritize in LIGHT mode.
_LIGHT_MODE_IMPORTANT_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "conclusion",
    "summary",
}

# Cap on how many section-level concept mentions we keep in LIGHT mode.
_RUNTIME_MAX_SECTION_CONCEPTS_LIGHT = 50


def _max_section_concepts_for_mode() -> Optional[int]:
    """
    Return a global cap on SectionConcept items produced by
    extract_concepts_from_sections, or None for 'no cap'.
    """
    if settings.runtime_mode == RuntimeMode.LIGHT:
        return _RUNTIME_MAX_SECTION_CONCEPTS_LIGHT
    # STANDARD and HEAVY: effectively no cap at this layer
    return None


# ---------------------------------------------------------------------------
# KeyBERT-backed concept extractor
# ---------------------------------------------------------------------------

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
          - paper.concepts (richer list of Concept objects)
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

        # Expose as list of (label, score) tuples
        paper.paper_level_concepts = [
            (c.label, c.weight) for c in paper_concept_objs
        ]
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

                # Attach to section in a flexible way
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


# ---------------------------------------------------------------------------
# Dataclasses for section-level concepts and aggregated summaries
# ---------------------------------------------------------------------------

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
    concept: Concept
    paper_id: Optional[str] = None
    section_title: Optional[str] = None
    section_level: Optional[int] = None


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


# ---------------------------------------------------------------------------
# Singleton extractor + low-level text API
# ---------------------------------------------------------------------------

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


def extract_concepts(
    text: str,
    top_n: Optional[int] = None,
) -> List[Concept]:
    """
    Global convenience wrapper around the singleton ConceptExtractor.

    Note: this is mode-agnostic. Runtime-aware behavior is applied in
    section/document helpers below.
    """
    extractor = get_concept_extractor()
    return extractor.extract_concepts(text, top_n=top_n)


# ---------------------------------------------------------------------------
# Section- / document-level concept extraction (runtime-aware)
# ---------------------------------------------------------------------------

def extract_concepts_from_section(
    section: PaperSection,
    paper_id: Optional[str] = None,
) -> List[SectionConcept]:
    """
    Extract concepts from a single PaperSection and wrap them as SectionConcepts.

    All runtime modes (LIGHT/STANDARD/HEAVY) perform extraction; LIGHT just
    truncates overall counts at a higher level.
    """
    if not getattr(section, "text", None):
        return []

    text = section.text or ""
    concepts = extract_concepts(text)

    results: List[SectionConcept] = []
    for c in concepts:
        results.append(
            SectionConcept(
                concept=c,
                paper_id=paper_id,
                section_title=getattr(section, "title", None) or None,
                section_level=getattr(section, "level", None),
            )
        )

    return results


def extract_concepts_from_sections(
    sections: Sequence[PaperSection],
    paper_id: Optional[str] = None,
) -> List[SectionConcept]:
    """
    Run concept extraction over each PaperSection.

    Runtime-aware behavior:
      - LIGHT   → run extraction but truncate total SectionConcepts to a cap.
      - STANDARD / HEAVY → full behavior.
    """
    results: List[SectionConcept] = []

    for sec in sections:
        results.extend(extract_concepts_from_section(sec, paper_id=paper_id))

    # LIGHT mode: trim total SectionConcepts if necessary
    max_concepts = _max_section_concepts_for_mode()
    if max_concepts is not None and len(results) > max_concepts:
        # Simple truncation preserves natural order (by section)
        results = results[:max_concepts]

    return results


def extract_concepts_from_document(
    doc: PaperDocument,
    config: Optional[Settings] = None,
) -> List[SectionConcept]:
    """
    Extract concepts from a PaperDocument, with behavior modulated by runtime_mode.

    STANDARD (default):
        - Process all sections.

    LIGHT:
        - Focus on the most important sections (abstract, intro, related work, conclusion).
        - If those can’t be identified, fall back to the first few sections.

    HEAVY:
        - For now, same as STANDARD. Later you might:
            * increase candidate limits
            * add extra passes (e.g., formula/concept extraction).
    """
    config = config or settings
    sections: Iterable[PaperSection] = doc.sections

    if config.runtime_mode == RuntimeMode.LIGHT:
        filtered = [
            s
            for s in sections
            if (
                getattr(s, "kind", None)
                and s.kind.lower() in _LIGHT_MODE_IMPORTANT_SECTIONS
            ) or (
                getattr(s, "title", None)
                and s.title.lower() in _LIGHT_MODE_IMPORTANT_SECTIONS
            )
        ]

        if filtered:
            sections_to_use = filtered
        else:
            all_sections = list(doc.sections)
            sections_to_use = all_sections[:3]
    else:
        # STANDARD and HEAVY both process all sections for now
        sections_to_use = list(doc.sections)

    concepts: List[SectionConcept] = []
    for section in sections_to_use:
        concepts.extend(extract_concepts_from_section(section, paper_id=getattr(doc, "paper_id", None)))

    max_concepts = _max_section_concepts_for_mode()
    if max_concepts is not None and len(concepts) > max_concepts:
        concepts = concepts[:max_concepts]

    return concepts


# ---------------------------------------------------------------------------
# Aggregation: SectionConcept -> ConceptSummary
# ---------------------------------------------------------------------------

def _default_section_weights() -> Dict[str, float]:
    """
    Default heuristic weights per section title (case-insensitive).
    """
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

    if title in weights:
        return weights[title]

    for key, w in weights.items():
        if key in title:
            return w

    return default_weight


def aggregate_section_concepts(
    section_concepts: Sequence[SectionConcept],
    section_weights: Optional[Mapping[str, float]] = None,
    default_section_weight: float = 1.0,
) -> Dict[str, ConceptSummary]:
    """
    Aggregate SectionConcept objects into per-concept summaries for a single paper.

    Returns:
        A dict mapping concept_key -> ConceptSummary.
    """
    if section_weights is None:
        section_weights = _default_section_weights()

    aggregates: Dict[str, Tuple[str, Optional[str], int, Dict[str, int], float]] = {}

    for sc in section_concepts:
        concept = sc.concept

        name = getattr(concept, "name", None) or getattr(concept, "label", None) or str(concept)
        kind = getattr(concept, "kind", None)

        concept_key = name

        section_title = sc.section_title or "UNKNOWN"
        weight = _lookup_section_weight(section_title, section_weights, default_section_weight)

        if concept_key not in aggregates:
            aggregates[concept_key] = (
                name,             # base_name
                kind,             # kind
                0,                # total_count
                defaultdict(int), # mentions_by_section
                0.0,              # weighted_score
            )

        base_name, concept_kind, total, per_section, weighted_score = aggregates[concept_key]

        total += 1
        per_section[section_title] += 1
        weighted_score += weight

        aggregates[concept_key] = (base_name, concept_kind, total, per_section, weighted_score)

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
