# kg_ai_papers/ingest/__init__.py

"""
Ingestion pipeline for going from raw PDFs -> TEI -> sections + concepts
and (optionally) persisting to a Neo4j graph store.
"""

from .pipeline import IngestedPaper, ingest_pdf, persist_ingested_paper_to_neo4j

__all__ = ["IngestedPaper", "ingest_pdf", "persist_ingested_paper_to_neo4j"]
