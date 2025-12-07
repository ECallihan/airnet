# kg_ai_papers/web/app.py

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional
import time
import asyncio
import logging
logger = logging.getLogger("kg_ai_papers.web")
logging.basicConfig(level=logging.INFO)

import networkx as nx
from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

from pydantic import BaseModel
from fastapi import UploadFile, File
from starlette.concurrency import run_in_threadpool
from kg_ai_papers.config.settings import settings
from kg_ai_papers.api.models import (
    PaperSummary,
    PaperInfluenceResult,
)
from kg_ai_papers.api.query import (
    get_paper_influence_view,
)
from kg_ai_papers.graph.storage import load_graph  # already there
from kg_ai_papers.graph.schema import NodeType

from kg_ai_papers.web.security import api_key_auth, rate_limiter

from pathlib import Path
from pydantic import BaseModel
from fastapi import UploadFile, File
from starlette.concurrency import run_in_threadpool

from kg_ai_papers.ingest.pipeline import (
    ingest_arxiv_paper,
    ingest_pdf_and_update_graph,
)
from kg_ai_papers.graph.storage import save_graph
from kg_ai_papers.graph.storage import save_graph
from kg_ai_papers.ingest.pipeline import (
    ingest_arxiv_paper,
    ingest_pdf_and_update_graph,
)



# -------------------------------------------------------------------
# Lifespan: load graph once at startup
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup:
      - Load the knowledge graph into memory
    On shutdown:
      - Drop the reference (GC will reclaim memory)
    """
    graph_name = getattr(settings, "GRAPH_DEFAULT_NAME", "graph")

    try:
        G = load_graph(name=graph_name)
    except FileNotFoundError:
        # You could decide to still start the app and 500 on usage
        G = nx.MultiDiGraph()

    app.state.graph = G

    # simple lock to serialize graph updates (ingestion)
    import asyncio
    app.state.graph_lock = asyncio.Lock()

    try:
        yield
    finally:
        app.state.graph = None
        app.state.graph_lock = None


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
    allow_origins=["*"],  # TODO: replace with your front-end origin once you have one
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class ArxivIngestRequest(BaseModel):
    arxiv_id: str

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _get_graph(app: FastAPI) -> nx.MultiDiGraph:
    G: nx.MultiDiGraph = app.state.graph
    if G is None:
        raise HTTPException(status_code=500, detail="Graph not loaded.")
    return G


def _iter_paper_nodes(G: nx.MultiDiGraph):
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

class ArxivIngestRequest(BaseModel):
    arxiv_id: str

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
    response_model=PaperInfluenceResult,
    summary="Get concepts and influence view for a paper",
)
async def get_paper(
    arxiv_id: str,
    top_k_concepts: int = Query(10, ge=1, le=100),
    top_k_references: int = Query(5, ge=0, le=100),
    top_k_influenced: int = Query(5, ge=0, le=100),
) -> PaperInfluenceResult:
    """
    Return a high-level influence view for the given paper:
      - its top concepts
      - the most influential references it builds on
      - the papers that most strongly build on it
    """
    G = _get_graph(app)

    try:
        result = get_paper_influence_view(
            G,
            arxiv_id=arxiv_id,
            top_k_concepts=top_k_concepts,
            top_k_references=top_k_references,
            top_k_influenced=top_k_influenced,
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Paper with arxiv_id={arxiv_id!r} not found in graph.",
        )

    return result

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
    lock = getattr(app_obj.state, "graph_lock", None)

    work_dir = settings.DATA_DIR / "ingest" / payload.arxiv_id
    work_dir.mkdir(parents=True, exist_ok=True)

    async def _do_ingest():
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
    lock = getattr(app_obj.state, "graph_lock", None)

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


@app.get(
    "/papers",
    response_model=List[PaperSummary],
    summary="List/search papers in the graph",
)
async def list_papers(
    q: Optional[str] = Query(
        None,
        description="Optional search query; matches in title or arxiv_id (case-insensitive).",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="Maximum number of papers to return.",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Offset into the result set (for pagination).",
    ),
) -> List[PaperSummary]:
    """
    List papers known to the graph.

    If `q` is provided, performs a simple case-insensitive substring search
    against titles and arxiv IDs.
    """
    G = _get_graph(app)

    papers: List[PaperSummary] = []

    q_lower = q.lower() if q else None

    for node_id, data in _iter_paper_nodes(G):
        arxiv_id = data.get("arxiv_id", "")
        title = data.get("title", "")

        if q_lower:
            if q_lower not in arxiv_id.lower() and q_lower not in title.lower():
                continue

        papers.append(
            PaperSummary(
                arxiv_id=arxiv_id,
                title=title,
                abstract=data.get("abstract"),
            )
        )

    # simple pagination
    papers = papers[offset : offset + limit]
    return papers

@app.post(
    "/ingest/arxiv",
    response_model=PaperInfluenceResult,
    summary="Ingest an arXiv paper, update the graph, and return its influence view",
)
async def ingest_arxiv(
    payload: ArxivIngestRequest,
    request: Request,
):
    """
    Ingest a single arXiv paper:

      1. Download the PDF from arXiv (via ingest_arxiv_paper).
      2. Run GROBID + TEI parsing + concept extraction.
      3. Update the in-memory NetworkX graph.
      4. Persist the updated graph to disk.
      5. Return the same influence view as GET /papers/{arxiv_id}.
    """
    app = request.app
    G = _get_graph(app)

    # Where to put GROBID output, TEI, etc. for this ingest
    work_dir = settings.DATA_DIR / "ingest" / payload.arxiv_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Optional concurrency protection
    lock = getattr(app.state, "graph_lock", None)
    if lock is not None:
        async with lock:
            ingested = await run_in_threadpool(
                ingest_arxiv_paper,
                arxiv_id=payload.arxiv_id,
                work_dir=work_dir,
            )
            # Update in-memory graph
            from kg_ai_papers.graph.builder import update_graph_with_ingested_paper
            update_graph_with_ingested_paper(G, ingested)

            # Persist to disk under the default graph name
            graph_name = getattr(settings, "GRAPH_DEFAULT_NAME", "graph")
            save_graph(G, name=graph_name)
    else:
        # Fallback if no lock is configured
        ingested = await run_in_threadpool(
            ingest_arxiv_paper,
            arxiv_id=payload.arxiv_id,
            work_dir=work_dir,
        )
        from kg_ai_papers.graph.builder import update_graph_with_ingested_paper
        update_graph_with_ingested_paper(G, ingested)
        graph_name = getattr(settings, "GRAPH_DEFAULT_NAME", "graph")
        save_graph(G, name=graph_name)

    # Return a fresh influence view from the updated graph
    try:
        result = get_paper_influence_view(
            G,
            arxiv_id=payload.arxiv_id,
            top_k_concepts=10,
            top_k_references=5,
            top_k_influenced=5,
        )
    except KeyError:
        # If ingestion somehow succeeded but the node isn't in the graph,
        # surface a 500 to make it obvious.
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
async def ingest_pdf_endpoint(
    request: Request,
    file: UploadFile = File(..., description="PDF file to ingest"),
):
    """
    Ingest an uploaded PDF and update the graph.

    The paper id is derived from the filename (stem) unless you extend this
    to accept an explicit arxiv_id or custom id.
    """
    app = request.app
    G = _get_graph(app)

    # Save uploaded PDF to a stable location
    uploads_dir = settings.DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = uploads_dir / file.filename
    content = await file.read()
    pdf_path.write_bytes(content)

    lock = getattr(app.state, "graph_lock", None)

    if lock is not None:
        async with lock:
            ingested = await run_in_threadpool(
                ingest_pdf_and_update_graph,
                G,
                pdf_path,
            )
            graph_name = getattr(settings, "GRAPH_DEFAULT_NAME", "graph")
            save_graph(G, name=graph_name)
    else:
        ingested = await run_in_threadpool(
            ingest_pdf_and_update_graph,
            G,
            pdf_path,
        )
        graph_name = getattr(settings, "GRAPH_DEFAULT_NAME", "graph")
        save_graph(G, name=graph_name)

    # IngestedPaperResult includes a Paper with arxiv_id (or filename stem)
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
