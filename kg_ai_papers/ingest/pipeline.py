from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests
import pickle

from kg_ai_papers.config.settings import settings
# Keep the import for type hints / compatibility, even if we don't rely on it.
from kg_ai_papers.grobid_client import GrobidClient  # type: ignore[unused-import]
from kg_ai_papers.tei_parser import extract_sections_from_tei
from kg_ai_papers.nlp.concept_extraction import (
    SectionConcept,
    ConceptSummary,
    extract_concepts_from_sections,
    aggregate_section_concepts,
)
from kg_ai_papers.models.paper import Paper


# -----------------------------------------------------------------------------
# Public result type for downstream consumers
# -----------------------------------------------------------------------------

@dataclass
class IngestedPaperResult:
    """
    Bundle the key outputs of ingestion for downstream consumers:

    - paper: core metadata (arxiv_id, title, abstract, year, etc.)
    - concept_summaries: aggregated concept scores for this paper
    - references: list of arxiv_ids this paper cites (outgoing edges)
      (currently optional / empty until citation extraction is wired in)
    """
    paper: Paper
    concept_summaries: Dict[str, ConceptSummary]
    references: List[str]


# -----------------------------------------------------------------------------
# Internal ingestion representation (PDF -> TEI -> concepts)
# -----------------------------------------------------------------------------

@dataclass
class IngestedPaper:
    """
    Result of ingesting a single paper from PDF through GROBID + concept extraction.

    This is deliberately lightweight and serialisable so it can be:
      - inspected in tests
      - persisted to disk
      - sent to a graph store like Neo4j
    """

    arxiv_id: str
    pdf_path: Path
    tei_path: Path
    sections: Sequence[Any]  # usually PaperSection, but kept loose for easier testing
    section_concepts: List[SectionConcept]
    concept_summaries: Dict[str, ConceptSummary]

# -----------------------------------------------------------------------------
# Low-level HTTP call to GROBID (fallback when no usable client is provided)
# -----------------------------------------------------------------------------

def _process_pdf_via_http(pdf_path: Path, work_dir: Optional[Path] = None) -> Path:
    """
    Send a PDF to GROBID /api/processFulltextDocument and return the TEI path.

    Uses environment variable GROBID_URL if set, otherwise settings.GROBID_URL.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Prefer an explicit env override, but fall back to configured settings.
    base_url = os.getenv("GROBID_URL", settings.GROBID_URL).rstrip("/")
    timeout = int(os.getenv("GROBID_TIMEOUT", "60"))

    # Choose a TEI output directory
    base_dir = work_dir if work_dir is not None else pdf_path.parent
    tei_dir = base_dir / "tei"
    tei_dir.mkdir(parents=True, exist_ok=True)

    # Optional health check
    try:
        r = requests.get(f"{base_url}/api/isalive", timeout=timeout)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"GROBID at {base_url!r} did not respond correctly to /api/isalive"
        ) from exc

    url = f"{base_url}/api/processFulltextDocument"
    files = {"input": pdf_path.open("rb")}
    params = {
        "consolidateHeader": 1,
        "consolidateCitations": 0,
        "segmentSentences": 0,
    }

    try:
        resp = requests.post(url, files=files, data=params, timeout=timeout)
    finally:
        files["input"].close()

    resp.raise_for_status()

    tei_xml = resp.text
    tei_path = tei_dir / f"{pdf_path.stem}.tei.xml"
    tei_path.write_text(tei_xml, encoding="utf-8")
    return tei_path

# -----------------------------------------------------------------------------
# Core ingestion: PDF -> TEI -> sections -> concepts
# -----------------------------------------------------------------------------

def ingest_pdf(
    pdf_path: Optional[Path] = None,
    *,
    arxiv_id: Optional[str] = None,
    grobid_client: Optional[Any] = None,
    neo4j_session: Optional[Any] = None,
    work_dir: Optional[Path] = None,
    use_cache: bool = True,
    force_reingest: bool = False,
    **_: Any,  # absorb unexpected kwargs for backwards compatibility
) -> IngestedPaper:
    """
    End-to-end ingestion for a single PDF.

    Pipeline:
        PDF -> (GROBID) -> TEI
             -> (tei_parser) -> sections
             -> (concept_extraction) -> SectionConcepts
             -> aggregate_section_concepts -> ConceptSummary per concept
             -> (optional) persist to Neo4j

    Call patterns supported:

        ingest_pdf(pdf_path, arxiv_id=..., grobid_client=...)
        ingest_pdf(arxiv_id=..., grobid_client=..., work_dir=...)
        ingest_pdf(pdf_path)  # everything inferred

    If pdf_path is not provided, we derive it as work_dir / f"{arxiv_id}.pdf".
    If that file does not exist, we try to download it from arXiv.

    Caching
    -------
    We cache the full IngestedPaper object to disk so repeated runs do not
    hit GROBID or the NLP stack again.

    Cache location (per paper):

        <base_dir>/cache/ingested/<paper_id>.pkl

    where base_dir is work_dir if given, else pdf_path.parent, and paper_id is
    arxiv_id if available, otherwise pdf_path.stem.

    Parameters
    ----------
    use_cache:
        If True (default), try to load from / write to cache.
        If False, never read or write cache (always recompute).
    force_reingest:
        If True, ignore existing cache and recompute. If use_cache is also
        True, the new result overwrites the old cache.
    """
    # Infer pdf_path if only arxiv_id + work_dir were given
    if pdf_path is None:
        if arxiv_id is None:
            raise ValueError(
                "ingest_pdf: either pdf_path must be provided, or "
                "both arxiv_id and work_dir must be given."
            )
        if work_dir is None:
            raise ValueError(
                "ingest_pdf: pdf_path is None and work_dir is None; cannot "
                "infer PDF path. Provide pdf_path or (arxiv_id + work_dir)."
            )
        pdf_path = Path(work_dir) / f"{arxiv_id}.pdf"

    pdf_path = Path(pdf_path)

    # If arxiv_id still isn't set, derive it from the file name
    if arxiv_id is None:
        arxiv_id = pdf_path.stem

    # Ensure the PDF actually exists; if not, try to download it from arXiv
    if not pdf_path.exists():
        if arxiv_id is not None:
            download_dir = work_dir if work_dir is not None else pdf_path.parent
            pdf_path = _download_arxiv_pdf(arxiv_id, Path(download_dir))
        else:
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ------------------------------------------------------------------
    # Ingestion cache
    # ------------------------------------------------------------------
    ingested: Optional[IngestedPaper] = None

    if use_cache:
        base_dir = work_dir if work_dir is not None else pdf_path.parent
        cache_dir = base_dir / "cache" / "ingested"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = arxiv_id or pdf_path.stem
        cache_path = cache_dir / f"{cache_key}.pkl"

        if cache_path.exists() and not force_reingest:
            try:
                with cache_path.open("rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, IngestedPaper) and loaded.arxiv_id == arxiv_id:
                    ingested = loaded
            except Exception:
                ingested = None
    else:
        cache_path = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # If no usable cache, run the full pipeline.
    # ------------------------------------------------------------------
    if ingested is None:
        # 1) PDF -> TEI via either an external GrobidClient or our HTTP fallback
        if grobid_client is not None:
            if hasattr(grobid_client, "parse_pdf_to_tei"):
                # New-style GrobidClient from kg_ai_papers.grobid_client
                tei_path = grobid_client.parse_pdf_to_tei(pdf_path)
            elif hasattr(grobid_client, "process_pdf"):
                # Backwards-compat for older clients
                tei_path = grobid_client.process_pdf(pdf_path)
            else:
                raise TypeError(
                    "grobid_client must implement parse_pdf_to_tei(pdf_path: Path) "
                    "or process_pdf(pdf_path: Path) -> Path"
                )
        else:
            tei_path = _process_pdf_via_http(pdf_path, work_dir=work_dir)

        # 2) TEI -> sections
        sections = extract_sections_from_tei(tei_path)

        # 3) sections -> per-section concepts
        section_concepts = extract_concepts_from_sections(
            sections,
            paper_id=arxiv_id,
        )

        # 4) aggregate to ConceptSummary per concept
        concept_summaries = aggregate_section_concepts(section_concepts)

        ingested = IngestedPaper(
            arxiv_id=arxiv_id,
            pdf_path=pdf_path,
            tei_path=tei_path,
            sections=list(sections),
            section_concepts=section_concepts,
            concept_summaries=concept_summaries,
        )

        # Persist cache for future runs
        if use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
                with cache_path.open("wb") as f:  # type: ignore[union-attr]
                    pickle.dump(ingested, f)
            except Exception:
                # Cache write failures should never break ingestion
                pass

    # 5) optional Neo4j persistence
    if neo4j_session is not None:
        persist_ingested_paper_to_neo4j(neo4j_session, ingested)

    return ingested

# -----------------------------------------------------------------------------
# Neo4j persistence
# -----------------------------------------------------------------------------

def persist_ingested_paper_to_neo4j(session: Any, ingested: IngestedPaper) -> None:
    """
    Persist an IngestedPaper into Neo4j.

    We keep the interface very generic: `session` is anything with a `.run`
    method (so we don't have to depend on the `neo4j` package in this project).

    Schema (high-level):

      (:Paper {arxiv_id, pdf_path, tei_path})
      (:Concept {key, name, kind})
      (:Paper)-[:MENTIONS {
          mentions_total,
          weighted_score,
          mentions_by_section
      }]->(:Concept)
    """
    session.run(
        """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        ON CREATE SET p.pdf_path = $pdf_path,
                      p.tei_path = $tei_path
        ON MATCH SET  p.pdf_path = $pdf_path,
                      p.tei_path = $tei_path
        """,
        arxiv_id= ingested.arxiv_id,
        pdf_path=str(ingested.pdf_path),
        tei_path=str(ingested.tei_path),
    )

    if not ingested.concept_summaries:
        return

    concepts_param = [
        {
            "concept_key": key,
            "name": summary.base_name,
            "kind": summary.kind,
            "mentions_total": summary.mentions_total,
            "weighted_score": summary.weighted_score,
            "mentions_by_section": summary.mentions_by_section,
        }
        for key, summary in ingested.concept_summaries.items()
    ]

    session.run(
        """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        UNWIND $concepts AS c
        MERGE (k:Concept {key: c.concept_key})
        ON CREATE SET k.name = c.name,
                      k.kind = c.kind
        MERGE (p)-[r:MENTIONS]->(k)
        SET r.mentions_total = c.mentions_total,
            r.weighted_score = c.weighted_score,
            r.mentions_by_section = c.mentions_by_section
        """,
        arxiv_id=ingested.arxiv_id,
        concepts=concepts_param,
    )

# -----------------------------------------------------------------------------
# Helper: download arXiv PDF into a work directory
# -----------------------------------------------------------------------------

def _download_arxiv_pdf(arxiv_id: str, work_dir: Path) -> Path:
    """
    Download the PDF for a given arXiv ID into work_dir and return its path.

    Uses the `arxiv` Python package.
    """
    import arxiv  # local import to avoid a hard dependency at import time

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    search = arxiv.Search(id_list=[arxiv_id])
    client = arxiv.Client()
    results = client.results(search)

    try:
        result = next(results)
    except StopIteration:
        raise RuntimeError(f"No arXiv result found for id {arxiv_id!r}")

    pdf_path = work_dir / f"{arxiv_id}.pdf"
    result.download_pdf(filename=str(pdf_path))
    return pdf_path

# -----------------------------------------------------------------------------
# Public convenience: ingest directly from arXiv ID
# -----------------------------------------------------------------------------

def ingest_arxiv_paper(
    *,
    arxiv_id: str,
    work_dir: Path,
    grobid_client: Optional[Any] = None,
    neo4j_session: Optional[Any] = None,
    references: Optional[List[str]] = None,
    use_cache: bool = True,
    force_reingest: bool = False,
) -> IngestedPaperResult:
    """
    High-level ingestion entrypoint for a paper identified by arxiv_id.

    This is the function your CLI calls:

        ingest_arxiv_paper(arxiv_id=..., work_dir=..., use_cache=..., force_reingest=...)

    Steps:
      1. Ensure the PDF for arxiv_id exists in work_dir (download if needed).
      2. Run ingest_pdf (which has its own cache for TEI + concepts).
      3. Build an IngestedPaperResult with a Paper model populated from arXiv
         metadata where possible.

    Flags
    -----
    use_cache:
        Passed through to ingest_pdf (see its docstring).
    force_reingest:
        Passed through to ingest_pdf (see its docstring).
    """
    work_dir = Path(work_dir)

    # 1) Ensure the PDF is present; don't re-download if it already exists
    pdf_path = work_dir / f"{arxiv_id}.pdf"
    if not pdf_path.exists():
        pdf_path = _download_arxiv_pdf(arxiv_id, work_dir)

    # 2) Run the core ingestion (this will use the IngestedPaper cache if present)
    ingested = ingest_pdf(
        pdf_path=pdf_path,
        arxiv_id=arxiv_id,
        grobid_client=grobid_client,
        neo4j_session=neo4j_session,
        work_dir=work_dir,
        use_cache=use_cache,
        force_reingest=force_reingest,
    )

    # 3) Fetch metadata from arXiv and build a Paper model
    title = ""
    abstract = ""

    try:
        import arxiv  # local import to keep optional

        search = arxiv.Search(id_list=[arxiv_id])
        client = arxiv.Client()
        results = client.results(search)

        try:
            result = next(results)
            title = result.title or ""
            abstract = result.summary or ""
        except StopIteration:
            pass
    except Exception:
        # Metadata fetch failure shouldn't kill ingestion; fall back to blanks
        pass

    paper = Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        pdf_path=str(pdf_path),
    )

    if references is None:
        references = []

    return IngestedPaperResult(
        paper=paper,
        concept_summaries=ingested.concept_summaries,
        references=references,
    )

# -----------------------------------------------------------------------------
# Optional convenience: ingest + update NetworkX graph in one call
# -----------------------------------------------------------------------------

def ingest_pdf_and_update_graph(
    G: Any,  # typically nx.MultiDiGraph; kept loose to avoid hard dependency here
    pdf_path: Path,
    *,
    grobid_client: Optional[Any] = None,
    neo4j_session: Optional[Any] = None,
    paper: Optional[Paper] = None,
    arxiv_id: Optional[str] = None,
    references: Optional[List[str]] = None,
) -> IngestedPaperResult:
    """
    Convenience helper: ingest a PDF, optionally persist to Neo4j, update the
    in-memory NetworkX graph, and return a high-level IngestedPaperResult.
    """
    pdf_path = Path(pdf_path)

    # Resolve the canonical id we will use everywhere
    if paper is not None:
        resolved_id = paper.arxiv_id
    elif arxiv_id is not None:
        resolved_id = arxiv_id
    else:
        resolved_id = pdf_path.stem

    # Run the core ingestion (PDF → TEI → sections → concepts [+ Neo4j])
    ingested = ingest_pdf(
        pdf_path,
        arxiv_id=resolved_id,
        grobid_client=grobid_client,
        neo4j_session=neo4j_session,
    )

    # If the caller didn't provide a Paper model, create a minimal placeholder.
    if paper is None:
        paper = Paper(arxiv_id=resolved_id)

    # References are optional / placeholder until TEI citation extraction or
    # arXiv metadata wiring is added.
    if references is None:
        references = []

    result = IngestedPaperResult(
        paper=paper,
        concept_summaries=ingested.concept_summaries,
        references=references,
    )

    # Update the in-memory graph using the standard helper.
    from kg_ai_papers.graph.builder import update_graph_with_ingested_paper
    update_graph_with_ingested_paper(G, result)

    return result
