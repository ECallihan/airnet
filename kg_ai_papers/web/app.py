# kg_ai_papers/web/app.py

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional
import time

import logging
logger = logging.getLogger("kg_ai_papers.web")
logging.basicConfig(level=logging.INFO)

import networkx as nx
from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

from kg_ai_papers.config.settings import settings
from kg_ai_papers.api.models import (
    PaperSummary,
    PaperInfluenceResult,
)
from kg_ai_papers.api.query import (
    get_paper_influence_view,
)
from kg_ai_papers.graph.storage import load_graph
from kg_ai_papers.graph.schema import NodeType

from kg_ai_papers.web.security import api_key_auth, rate_limiter


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
    yield
    app.state.graph = None


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
