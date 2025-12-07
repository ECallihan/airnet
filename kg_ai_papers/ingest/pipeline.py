# kg_ai_papers/ingest/pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from kg_ai_papers.grobid_client import GrobidClient
from kg_ai_papers.tei_parser import extract_sections_from_tei
from kg_ai_papers.nlp.concept_extraction import (
    SectionConcept,
    ConceptSummary,
    extract_concepts_from_sections,
    aggregate_section_concepts,
)
from kg_ai_papers.models.paper import Paper


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


@dataclass
class IngestedPaper:
    """
    Result of ingesting a single paper from PDF through GROBID + concept extraction.

    This is deliberately lightweight and serialisable so it can be:
      - inspected in tests
      - persisted to disk
      - sent to a graph store like Neo4j
    """

    paper_id: str
    pdf_path: Path
    tei_path: Path
    sections: Sequence[Any]  # usually PaperSection, but kept loose for easier testing
    section_concepts: List[SectionConcept]
    concept_summaries: Dict[str, ConceptSummary]


def ingest_pdf(
    pdf_path: Path,
    *,
    paper_id: Optional[str] = None,
    grobid_client: GrobidClient,
    neo4j_session: Optional[Any] = None,
) -> IngestedPaper:
    """
    End-to-end ingestion for a single PDF:

      PDF -> (GROBID) -> TEI
          -> (tei_parser) -> sections
          -> (concept_extraction) -> section concepts + aggregated concept summaries
          -> (optional) persist to Neo4j

    Parameters
    ----------
    pdf_path:
        Path to the input PDF.
    paper_id:
        Logical identifier for the paper (e.g. arXiv ID). If omitted, uses the
        PDF stem (filename without extension).
    grobid_client:
        Instance of GrobidClient (or a drop-in with .process_pdf(Path) -> Path).
    neo4j_session:
        Optional Neo4j session-like object with a .run(cypher: str, **params)
        method. If provided, the ingested paper will be persisted to Neo4j.

    Returns
    -------
    IngestedPaper
        Aggregated representation of the paper and its concepts.
    """
    if paper_id is None:
        paper_id = pdf_path.stem

    # 1) PDF -> TEI via GROBID
    tei_path = grobid_client.process_pdf(pdf_path)

    # 2) TEI -> sections
    sections = extract_sections_from_tei(tei_path)

    # 3) sections -> per-section concepts
    section_concepts = extract_concepts_from_sections(sections, paper_id=paper_id)

    # 4) aggregate to ConceptSummary per concept
    concept_summaries = aggregate_section_concepts(section_concepts)

    ingested = IngestedPaper(
        paper_id=paper_id,
        pdf_path=pdf_path,
        tei_path=tei_path,
        sections=list(sections),
        section_concepts=section_concepts,
        concept_summaries=concept_summaries,
    )

    # 5) optional Neo4j persistence
    if neo4j_session is not None:
        persist_ingested_paper_to_neo4j(neo4j_session, ingested)

    return ingested


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

    This focuses on paper <-> concept relationships; citation/influence edges
    can be added later using the existing NetworkX graph as ground truth.
    """
    # 1) Upsert the Paper node
    session.run(
        """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        ON CREATE SET p.pdf_path = $pdf_path,
                      p.tei_path = $tei_path
        ON MATCH SET  p.pdf_path = $pdf_path,
                      p.tei_path = $tei_path
        """,
        arxiv_id=ingested.paper_id,
        pdf_path=str(ingested.pdf_path),
        tei_path=str(ingested.tei_path),
    )

    if not ingested.concept_summaries:
        return

    # 2) Upsert Concept nodes + MENTIONS relationships
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
        arxiv_id=ingested.paper_id,
        concepts=concepts_param,
    )


# --- Optional convenience: ingest + update NetworkX graph --------------------


def ingest_pdf_and_update_graph(
    G: "Any",  # typically nx.MultiDiGraph; kept loose to avoid hard dependency here
    pdf_path: Path,
    *,
    grobid_client: GrobidClient,
    neo4j_session: Optional[Any] = None,
    paper: Optional[Paper] = None,
    paper_id: Optional[str] = None,
    references: Optional[List[str]] = None,
) -> IngestedPaperResult:
    """
    Convenience helper: ingest a PDF, optionally persist to Neo4j, update the
    in-memory NetworkX graph, and return a high-level IngestedPaperResult.

    Parameters
    ----------
    G:
        A NetworkX MultiDiGraph (or compatible) representing the in-memory KG.
    pdf_path:
        Path to the input PDF.
    grobid_client:
        Grobid client instance used by `grobid_client.process_pdf`.
    neo4j_session:
        Optional Neo4j session; passed through to `ingest_pdf`.
    paper:
        Optional pre-built Paper model. If provided, its `arxiv_id` drives
        the paper_id used throughout.
    paper_id:
        Optional explicit paper_id / arxiv_id. If both `paper` and `paper_id`
        are None, we fall back to `pdf_path.stem`.
    references:
        Optional list of arxiv_ids this paper cites. For now this usually
        remains empty until citation extraction is wired in.

    Returns
    -------
    IngestedPaperResult
        High-level bundle with Paper + concept summaries + references.
    """
    # Resolve the canonical id we will use everywhere
    if paper is not None:
        resolved_id = paper.arxiv_id
    elif paper_id is not None:
        resolved_id = paper_id
    else:
        resolved_id = pdf_path.stem

    # Run the core ingestion (PDF → TEI → sections → concepts [+ Neo4j])
    ingested = ingest_pdf(
        pdf_path,
        paper_id=resolved_id,
        grobid_client=grobid_client,
        neo4j_session=neo4j_session,
    )

    # If the caller didn't provide a Paper model, create a minimal placeholder.
    if paper is None:
        paper = Paper(arxiv_id=resolved_id)

    # References are currently optional / placeholder until TEI citation
    # extraction or arXiv metadata wiring is added.
    if references is None:
        references = []

    result = IngestedPaperResult(
        paper=paper,
        concept_summaries=ingested.concept_summaries,
        references=references,
    )

    # Update the in-memory graph using the standard helper.
    # Local import to avoid import-time cycles.
    from kg_ai_papers.graph.builder import update_graph_with_ingested_paper

    update_graph_with_ingested_paper(G, result)

    return result
