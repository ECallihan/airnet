# kg_ai_papers/parsing/pipeline.py

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

import numpy as np

from kg_ai_papers.config.settings import settings
from kg_ai_papers.models.paper import Paper
from kg_ai_papers.nlp.concept_extraction import get_concept_extractor
from kg_ai_papers.nlp.embedding import get_embedding_model
from kg_ai_papers.parsing.grobid_client import GrobidClientError, process_fulltext
from kg_ai_papers.parsing.tei_parser import parse_references, parse_sections


# ---------------------------------------------------------------------------
# Parsing (PDF -> sections + references)
# ---------------------------------------------------------------------------


def parse_pdf_to_paper(paper: Paper, pdf_path: Optional[str] = None) -> None:
    """
    Given a Paper with metadata and a PDF path, enrich it with sections & references
    using Grobid. If Grobid fails, we log via GrobidClientError and leave sections/
    references empty (the caller should catch and handle).
    """
    pdf_path = pdf_path or paper.pdf_path
    if not pdf_path:
        # Nothing we can do
        paper.sections = []
        paper.references = []
        return

    try:
        tei_xml = process_fulltext(pdf_path)
    except GrobidClientError:
        # Let the caller decide how to warn/log; for safety, ensure these attrs exist.
        paper.sections = []
        paper.references = []
        raise

    sections = parse_sections(tei_xml)
    references = parse_references(tei_xml)

    paper.sections = sections
    paper.references = references


# ---------------------------------------------------------------------------
# NLP enrichment (concepts + embeddings)
# ---------------------------------------------------------------------------


def enrich_paper(paper: Paper) -> None:
    """
    Run concept extraction + embedding for a Paper in-place.

    - paper.paper_level_concepts: list[(label, score)]
    - paper.concepts: richer Concept objects
    - paper.embedding: embedding vector (torch tensor or numpy/list, depending on embedder)
    """
    ce = get_concept_extractor()
    ce.enrich_paper(paper)

    embedder = get_embedding_model()
    paper.embedding = embedder.encode_paper(paper)


# ---------------------------------------------------------------------------
# JSON (de)serialization for enriched papers
# ---------------------------------------------------------------------------
def _to_serializable(obj):
    """
    Recursively convert Pydantic models, dataclasses, and other nested objects
    into JSON-serializable structures (dicts, lists, primitives).
    """
    # Pydantic model
    if isinstance(obj, BaseModel):
        # Works for v1 (.dict) and v2 (.model_dump)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()

    # Dataclass
    if is_dataclass(obj):
        return asdict(obj)

    # Dict
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    # List / tuple / set
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]

    # Everything else: assume it's already JSON-serializable (str, int, float, None, etc.)
    return obj


def _paper_to_serializable_dict(paper: Paper) -> dict:
    """
    Convert a Paper to a JSON-serializable dict without relying solely on .model_dump().

    Supports:
    - Pydantic v1/v2 models (via .dict() or .model_dump())
    - dataclasses (via asdict)
    - plain objects (via __dict__)
    Also normalizes the embedding to a list of floats and recursively converts
    nested models (Concept, ConceptOccurrence, Section, etc.).
    """
    # Base mapping for the paper itself
    if hasattr(paper, "model_dump"):
        data = paper.model_dump()
    elif hasattr(paper, "dict"):
        data = paper.dict()  # type: ignore[call-arg]
    elif is_dataclass(paper):
        data = asdict(paper)
    else:
        data = dict(paper.__dict__)

    # Normalize embedding on the original object first (so we don't miss it)
    emb = getattr(paper, "embedding", None)
    if emb is not None:
        try:
            try:
                import torch  # type: ignore

                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().tolist()
                else:
                    emb = np.asarray(emb, dtype=float).tolist()
            except ImportError:
                # No torch: just use numpy
                emb = np.asarray(emb, dtype=float).tolist()
        except Exception:
            emb = None

    # Make sure the dict has the cleaned embedding
    if emb is not None:
        data["embedding"] = emb
    else:
        data["embedding"] = None

    # Recursively convert all nested objects (Concept, ConceptOccurrence, etc.)
    data = _to_serializable(data)

    return data


def save_enriched_paper(paper: Paper) -> Path:
    """
    Save an enriched Paper (with concepts + embedding) to JSON under data/enriched/.
    """
    settings.enriched_dir.mkdir(parents=True, exist_ok=True)
    path = settings.enriched_dir / f"{paper.arxiv_id}.json"

    data = _paper_to_serializable_dict(paper)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return path


def load_enriched_paper(path: Path) -> Paper:
    """
    Load a Paper from an enriched JSON file.
    Embedding is stored as a list[float]; we leave it as-is and let downstream
    code (e.g., graph builder) convert to numpy/torch as needed.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Simple reconstruction; Paper is flexible enough to accept this
    paper = Paper(**data)
    return paper


# ---------------------------------------------------------------------------
# High-level helper used by the CLI pipeline
# ---------------------------------------------------------------------------


def ensure_enriched(
    paper: Paper,
    pdf_path: Optional[str] = None,
    overwrite: bool = False,
) -> Paper:
    """
    Ensure that an enriched JSON exists for this paper and return an enriched Paper.
    """
    settings.enriched_dir.mkdir(parents=True, exist_ok=True)
    path = settings.enriched_dir / f"{paper.arxiv_id}.json"

    # Use explicit pdf_path if provided, else fall back to paper.pdf_path
    effective_pdf_path = pdf_path or paper.pdf_path

    if path.exists() and not overwrite:
        # Try to load existing enriched JSON; if it's corrupt or empty,
        # fall through and regenerate it.
        try:
            return load_enriched_paper(path)
        except Exception:
            pass

    # (Re-)parse and enrich
    try:
        parse_pdf_to_paper(paper, pdf_path=effective_pdf_path)
    except GrobidClientError:
        # Grobid failure: leave sections/references empty but continue
        pass

    enrich_paper(paper)
    save_enriched_paper(paper)
    return paper
