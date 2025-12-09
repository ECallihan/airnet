from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from networkx import MultiDiGraph
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from dataclasses import asdict, is_dataclass

from kg_ai_papers.api.models import (
    PaperDetail as ApiPaperDetail,
    PaperInfluenceResult,
    PaperSummary as ApiPaperSummary,
)
from kg_ai_papers.api.query import get_paper_influence_view

from kg_ai_papers.config.settings import settings
from kg_ai_papers.graph.schema import NodeType
from kg_ai_papers.graph.storage import save_graph, load_latest_graph
from kg_ai_papers.ingest.pipeline import ingest_arxiv_paper, ingest_pdf_and_update_graph
from kg_ai_papers.web.security import api_key_auth, rate_limiter

logger = logging.getLogger("kg_ai_papers.web")
logging.basicConfig(level=logging.INFO)


# -------------------------------------------------------------------
# Lifespan: load graph once at startup
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown handler:
    - Load the latest persisted graph (if any)
    - Otherwise start with an empty MultiDiGraph

    No Neo4j integration yet (placeholder for future expansion).
    """
    try:
        G = load_latest_graph()
    except Exception:
        logger.exception("Failed to load persisted graph; starting with a fresh graph")
        G = None

    if G is None:
        logger.info("No existing graph found; starting with fresh in-memory graph")
        G = nx.MultiDiGraph()

    app.state.graph = G
    app.state.neo4j_driver = None  # placeholder for future use

    yield

    # No shutdown actions yet


app = FastAPI(
    title="AI Paper Knowledge Graph API",
    description="API for querying concepts and influence between AI research papers.",
    version="0.1.0",
    lifespan=lifespan,
)

# -------------------------------------------------------------------
# CORS – allow everything for now (tighten later)
# -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to front-end origin later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Middleware
# -------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log method, path, and response status. Simple but effective.
    """
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    start = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start) * 1000
    logger.info(
        f"Completed {request.method} {request.url.path} "
        f"with status {response.status_code} in {duration_ms:.2f}ms"
    )

    return response


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------


class ArxivIngestRequest(BaseModel):
    arxiv_id: str


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _get_graph(app_obj: FastAPI) -> MultiDiGraph:
    """
    Fetch the in-memory graph from app.state, initializing if needed.
    """
    G = getattr(app_obj.state, "graph", None)
    if G is None:
        G = MultiDiGraph()
        app_obj.state.graph = G
    return G


def _iter_paper_nodes(G: nx.MultiDiGraph):
    """
    Yield (node_id, data) for nodes that are typed as papers.
    """
    for node_id, data in G.nodes(data=True):
        if data.get("type") == NodeType.PAPER.value:
            yield node_id, data


def _get_neo4j_session_or_none():
    """
    Best-effort Neo4j session factory.

    Uses environment variables:
      - NEO4J_URI
      - NEO4J_USER
      - NEO4J_PASSWORD

    If any are missing, or the driver is unavailable, returns None.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not user or not password:
        return None

    try:
        from neo4j import GraphDatabase  # type: ignore[import]
    except ImportError:
        logger.warning("Neo4j driver not installed; skipping Neo4j persistence.")
        return None

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return driver.session()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to create Neo4j session: %s", exc)
        return None


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------


@app.get("/health", summary="Health check")
async def health() -> dict:
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.get(
    "/papers/{arxiv_id}",
    response_model=ApiPaperDetail,
    summary="Get a single paper and its influence view",
)
@app.get(
    "/papers/{arxiv_id}",
    response_model=Dict[str, Any],
    summary="Get a single paper and its influence view",
)
async def get_paper(
    arxiv_id: str,
    request: Request,
    top_k_concepts: int = 10,
    top_k_references: int = 10,
    top_k_influenced: int = 10,
) -> Dict[str, Any]:
    """
    Return a detailed view of a single paper, including its metadata and
    influence information derived from the in-memory graph.

    - 404 if the paper is not present in the graph.
    """
    G = _get_graph(request.app)

    # Locate the paper metadata node
    paper_meta: Optional[Dict[str, Any]] = None
    for _, data in G.nodes(data=True):
        if data.get("arxiv_id") == arxiv_id:
            paper_meta = data
            break

    if paper_meta is None:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")

    # Compute the influence view (dataclass-ish object) from the graph
    influence_view = get_paper_influence_view(
        G,
        arxiv_id=arxiv_id,
        top_k_concepts=top_k_concepts,
        top_k_references=top_k_references,
        top_k_influenced=top_k_influenced,
    )

    # Convert the view into a plain dict so FastAPI/Pydantic can serialize it,
    # and so tests can access top-level "paper" and "concepts" keys.
    if isinstance(influence_view, dict):
        payload: Dict[str, Any] = dict(influence_view)
    elif is_dataclass(influence_view):
        payload = asdict(influence_view)
    elif hasattr(influence_view, "model_dump"):
        # Just in case we ever swap to a Pydantic model here
        payload = influence_view.model_dump()
    else:
        # Fallback: wrap minimal information
        payload = {
            "paper": {
                "arxiv_id": arxiv_id,
                "title": paper_meta.get("title") or "",
            },
            "concepts": [],
        }

    # Ensure the paper metadata includes the abstract from the graph, if present.
    paper_dict = payload.setdefault("paper", {})
    paper_dict.setdefault("arxiv_id", arxiv_id)
    paper_dict.setdefault("title", paper_meta.get("title") or "")
    if "abstract" not in paper_dict:
        paper_dict["abstract"] = paper_meta.get("abstract")

    return payload


@app.get(
    "/papers",
    response_model=List[ApiPaperSummary],
    summary="List all papers currently in the graph",
)
async def list_papers(request: Request) -> List[ApiPaperSummary]:
    """
    List all papers in the current in-memory graph.

    We treat any node with an `arxiv_id` attribute as a paper node and
    return a minimal summary for each.
    """
    G = _get_graph(request.app)

    summaries: List[ApiPaperSummary] = []
    for _, data in G.nodes(data=True):
        arxiv_id = data.get("arxiv_id")
        if not arxiv_id:
            continue

        summaries.append(
            ApiPaperSummary(
                arxiv_id=arxiv_id,
                title=data.get("title") or "",
                abstract=data.get("abstract") or "",
            )
        )

    return summaries


@app.post(
    "/ingest/arxiv",
    response_model=PaperInfluenceResult,
    summary="Ingest an arXiv paper, update the graph, and return its influence view",
)
async def ingest_arxiv(
    payload: ArxivIngestRequest,
    request: Request,
) -> PaperInfluenceResult:
    """
    Ingest a single arXiv paper:

      1. Download the PDF from arXiv.
      2. Run GROBID + TEI parsing + concept extraction.
      3. Update the in-memory NetworkX graph.
      4. Persist the updated graph to disk.
      5. Return the same influence view as GET /papers/{arxiv_id}.
    """
    app_obj = request.app
    G = _get_graph(app_obj)
    lock: Optional[asyncio.Lock] = getattr(app_obj.state, "graph_lock", None)

    work_dir = settings.DATA_DIR / "ingest" / payload.arxiv_id
    work_dir.mkdir(parents=True, exist_ok=True)

    async def _do_ingest() -> None:
        neo4j_session = _get_neo4j_session_or_none()
        try:
            ingested = await run_in_threadpool(
                ingest_arxiv_paper,
                arxiv_id=payload.arxiv_id,
                work_dir=work_dir,
                neo4j_session=neo4j_session,
            )
        finally:
            if neo4j_session is not None and hasattr(neo4j_session, "close"):
                neo4j_session.close()

        # Update in-memory graph
        from kg_ai_papers.graph.builder import update_graph_with_ingested_paper

        update_graph_with_ingested_paper(G, ingested)

        # Persist updated graph
        graph_name = getattr(settings, "GRAPH_DEFAULT_NAME", "graph")
        save_graph(G, name=graph_name)

    if lock is not None:
        async with lock:
            await _do_ingest()
    else:
        await _do_ingest()

    # Return a fresh influence view from updated graph
    try:
        result = get_paper_influence_view(
            G,
            arxiv_id=payload.arxiv_id,
            top_k_concepts=10,
            top_k_references=5,
            top_k_influenced=5,
        )
    except KeyError:
        raise HTTPException(
            status_code=500,
            detail=f"Ingested paper {payload.arxiv_id!r} not found in graph after update.",
        )

    return result


@app.post(
    "/ingest/pdf",
    response_model=PaperInfluenceResult,
    summary="Upload a PDF, ingest it, update the graph, and return its influence view",
)
async def ingest_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to ingest"),
) -> PaperInfluenceResult:
    """
    Ingest an uploaded PDF and update the graph.

    The logical paper id is derived from the filename stem unless the
    ingestion pipeline infers/sets a more specific arxiv_id.
    """
    app_obj = request.app
    G = _get_graph(app_obj)
    lock: Optional[asyncio.Lock] = getattr(app_obj.state, "graph_lock", None)

    uploads_dir = settings.DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = uploads_dir / file.filename
    content = await file.read()
    pdf_path.write_bytes(content)

    async def _do_ingest():
        neo4j_session = _get_neo4j_session_or_none()
        try:
            ingested = await run_in_threadpool(
                ingest_pdf_and_update_graph,
                G,
                pdf_path,
                neo4j_session=neo4j_session,
            )
        finally:
            if neo4j_session is not None and hasattr(neo4j_session, "close"):
                neo4j_session.close()

        # Persist updated graph
        graph_name = getattr(settings, "GRAPH_DEFAULT_NAME", "graph")
        save_graph(G, name=graph_name)

        return ingested

    if lock is not None:
        async with lock:
            ingested = await _do_ingest()
    else:
        ingested = await _do_ingest()

    # IngestedPaperResult.paper.arxiv_id is our logical id
    arxiv_id = ingested.paper.arxiv_id

    try:
        result = get_paper_influence_view(
            G,
            arxiv_id=arxiv_id,
            top_k_concepts=10,
            top_k_references=5,
            top_k_influenced=5,
        )
    except KeyError:
        raise HTTPException(
            status_code=500,
            detail=f"Ingested PDF for {arxiv_id!r} not found in graph after update.",
        )

    return result
