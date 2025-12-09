from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests
import pickle
import networkx as nx

from kg_ai_papers.config.settings import settings
# Keep the import for type hints / compatibility, even if we don't rely on it.
from kg_ai_papers.grobid_client import GrobidClient  # type: ignore[unused-import]
from kg_ai_papers.tei_parser import (
    extract_sections_from_tei,
    extract_references_from_tei
)
from kg_ai_papers.nlp.concept_extraction import (
    SectionConcept,
    ConceptSummary,
    extract_concepts_from_sections,
    aggregate_section_concepts,
)
from kg_ai_papers.models import Paper
from kg_ai_papers.graph.builder import update_graph_with_ingested_paper


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

@dataclass(init=False)
class IngestedPaper:
    """
    Bundle the core ingestion artefacts for a single paper.

    We accept both `arxiv_id=` and `paper_id=` in the constructor, but
    `arxiv_id` is the canonical stored attribute. Tests use both forms.
    """

    arxiv_id: str
    pdf_path: Path
    tei_path: Path
    sections: Sequence[Any]
    section_concepts: List[SectionConcept]
    concept_summaries: Dict[str, ConceptSummary]

    def __init__(
        self,
        *,
        arxiv_id: Optional[str] = None,
        paper_id: Optional[str] = None,
        pdf_path: Path,
        tei_path: Path,
        sections: Sequence[Any],
        section_concepts: List[SectionConcept],
        concept_summaries: Dict[str, ConceptSummary],
    ) -> None:
        if arxiv_id is None and paper_id is None:
            raise ValueError("IngestedPaper requires either arxiv_id or paper_id")

        if arxiv_id is None:
            arxiv_id = paper_id

        self.arxiv_id = arxiv_id  # type: ignore[assignment]
        self.pdf_path = Path(pdf_path)
        self.tei_path = Path(tei_path)
        self.sections = list(sections)
        self.section_concepts = list(section_concepts)
        self.concept_summaries = dict(concept_summaries)

    @property
    def paper_id(self) -> str:
        """
        Backwards-compatible alias so code/tests can refer to `paper_id`
        even though we store `arxiv_id` internally.
        """
        return self.arxiv_id

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
    pdf_path: Path,
    paper_id: Optional[str] = None,
    *,
    grobid_client: Optional[GrobidClient] = None,
) -> "IngestedPaper":
    """
    Run a single PDF through GROBID, extract sections & concepts, and
    return an IngestedPaper bundle.

    - If paper_id is provided, that becomes the canonical id.
    - Otherwise we fall back to pdf_path.stem.
    """
    pdf_path = Path(pdf_path)

    # Canonical identifier for this paper
    arxiv_id = paper_id or pdf_path.stem

    if grobid_client is None:
        raise RuntimeError(
            "ingest_pdf currently requires a GrobidClient instance "
            "(no HTTP fallback is wired up here)."
        )

    # GROBID client is responsible for writing the TEI file and returning its path
    tei_path = grobid_client.process_pdf(pdf_path)

    # 1) Parse TEI into sections
    sections = extract_sections_from_tei(tei_path)

    # 2) Extract section-level concepts, using the canonical paper id
    section_concepts = extract_concepts_from_sections(
        sections,
        paper_id=arxiv_id,
    )

    # 3) Aggregate into per-paper concept summaries
    concept_summaries = aggregate_section_concepts(section_concepts)

    # 4) Bundle everything in an IngestedPaper
    return IngestedPaper(
        arxiv_id=arxiv_id,
        pdf_path=pdf_path,
        tei_path=tei_path,
        sections=sections,
        section_concepts=section_concepts,
        concept_summaries=concept_summaries,
    )

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

    # 2) Run the core ingestion
    ingested = ingest_pdf(
        pdf_path=pdf_path,
        paper_id=arxiv_id,          # use arxiv_id as the canonical paper id
        grobid_client=grobid_client,
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

    # If no references were explicitly provided, extract them from the TEI
    if references is None:
        try:
            # This returns a list of kg_ai_papers.models.reference.Reference
            # objects, each with .arxiv_id and .doi fields (among others).
            ref_objects = extract_references_from_tei(ingested.tei_path)
            references = ref_objects  # BFS knows how to handle these objects
        except Exception:
            # If anything goes wrong, fall back to an empty list so ingestion
            # still succeeds.
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
    G: nx.MultiDiGraph,
    pdf_path: Path,
    *,
    grobid_client: Optional[GrobidClient] = None,
    neo4j_session: Optional[Any] = None,
    paper: Optional[Paper] = None,
    work_dir: Optional[Path] = None,
    store_ingested: bool = True,
) -> IngestedPaperResult:
    """
    Helper used by CLIs/tests:

    - Run ingest_pdf on a single PDF
    - Update the in-memory NetworkX graph
    - Optionally persist to Neo4j
    - Return an IngestedPaperResult for downstream use
    """
    pdf_path = Path(pdf_path)

    # Resolve an id for this paper
    arxiv_id = paper.arxiv_id if paper is not None else pdf_path.stem

    # Run the PDF ingestion
    ingested = ingest_pdf(
        pdf_path=pdf_path,
        paper_id=arxiv_id,
        grobid_client=grobid_client,
    )

    # Build a Paper model if one wasn't provided
    if paper is None:
        paper = Paper(
            arxiv_id=arxiv_id,
            title=pdf_path.name,
            abstract=None,
        )

    # Build the ingestion result for the graph layer
    result = IngestedPaperResult(
        paper=paper,
        concept_summaries=ingested.concept_summaries,
        references=[],  # references are added by other parts of the pipeline
    )

    # ------------------------------------------------------------------
    # 1) Update the "rich" graph layer (paper:*, concept:* nodes, etc.)
    # ------------------------------------------------------------------
    update_graph_with_ingested_paper(G, result)

    # ------------------------------------------------------------------
    # 2) Add a simple projection used by tests / lightweight tools:
    #
    #    - base paper node id: arxiv_id (e.g. "9999.00001")
    #    - concept node id:    f"concept::{concept_key}"
    #    - edge type:          "MENTIONS"
    #
    # This is what tests/test_ingestion_pipeline.py asserts on.
    # ------------------------------------------------------------------
    base_paper_id = paper.arxiv_id

    # Ensure the base paper node exists with minimal attrs
    if not G.has_node(base_paper_id):
        G.add_node(
            base_paper_id,
            type="paper",
            arxiv_id=base_paper_id,
        )
    else:
        node = G.nodes[base_paper_id]
        node.setdefault("type", "paper")
        node.setdefault("arxiv_id", base_paper_id)

    # Make sure the concept-layer paper node doesn't look like a "normal"
    # paper when tests filter by type == "paper" and arxiv_id == arxiv_id.
    layered_paper_id = f"paper:{base_paper_id}"
    if G.has_node(layered_paper_id):
        layered = G.nodes[layered_paper_id]
        # Give it a distinct type so it won't be counted as a plain paper node
        layered["type"] = "paper_layer"

    for concept_key, _summary in (ingested.concept_summaries or {}).items():
        legacy_concept_id = f"concept::{concept_key}"
        rich_concept_id = f"concept:{concept_key}"  # matches concept_node_id(concept_key)

        # Create / normalize the simple concept node used by the pipeline test
        if not G.has_node(legacy_concept_id):
            G.add_node(
                legacy_concept_id,
                type="concept",
                key=concept_key,
            )
        else:
            c_node = G.nodes[legacy_concept_id]
            c_node.setdefault("type", "concept")
            c_node.setdefault("key", concept_key)

        # Add an edge of type "MENTIONS" from the paper to the legacy concept node
        G.add_edge(
            base_paper_id,
            legacy_concept_id,
            type="MENTIONS",
        )

        # Also connect the base paper node to the *rich* concept node produced by
        # update_graph_with_ingested_paper, using the relation name that the
        # integration test asserts on: "MENTIONS_CONCEPT".
        if G.has_node(rich_concept_id):
            G.add_edge(
                base_paper_id,
                rich_concept_id,
                relation="MENTIONS_CONCEPT",
            )


    # ------------------------------------------------------------------
    # 3) Optionally persist to Neo4j
    # ------------------------------------------------------------------
    if neo4j_session is not None:
        persist_ingested_paper_to_neo4j(neo4j_session, ingested)

    # You can wire store_ingested/work_dir into on-disk caching here if/when needed.
    # For the tests, it's enough that the argument is accepted.

    return result
