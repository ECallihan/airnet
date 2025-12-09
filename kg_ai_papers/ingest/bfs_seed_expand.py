# kg_ai_papers/ingest/bfs_seed_expand.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from kg_ai_papers.grobid_client import GrobidClient
from kg_ai_papers.ingest.pipeline import ingest_arxiv_paper, IngestedPaperResult
from kg_ai_papers.graph.builder import update_graph_with_ingested_paper
from kg_ai_papers.graph.storage import load_latest_graph, save_graph


@dataclass
class SeedPaper:
    """
    Seed for BFS ingestion.

    For now we primarily ingest by arxiv_id. DOI is kept so that if/when you
    add DOI-based ingestion, de-duplication will still work.
    """
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None


@dataclass
class BFSConfig:
    """
    Configuration for breadth-first ingestion.

    max_papers : hard cap on the total number of papers ingested
    max_depth  : BFS depth (0 = seeds, 1 = their references, etc.)
    use_cache  : whether to reuse ingest cache from previous runs
    work_dir   : where PDFs/TEI/cache live for this run
    start_fresh: if True, ignore any existing graph and start from empty
    """
    max_papers: int = 200
    max_depth: int = 2
    use_cache: bool = True
    work_dir: Path = Path("data/bfs_ingest")
    start_fresh: bool = False


# ---------------------------------------------------------------------------
# Reference ID extraction (tightened)
# ---------------------------------------------------------------------------

def _seen_key(arxiv_id: Optional[str], doi: Optional[str]) -> Optional[str]:
    """
    Normalize an identifier pair (arxiv_id, doi) into a single key
    for de-duplication.

    Preference order:
      1. arxiv_id (if present)
      2. doi (lowercased)
    """
    if arxiv_id:
        return f"arxiv:{arxiv_id.strip()}"
    if doi:
        return f"doi:{doi.strip().lower()}"
    return None


def _iter_reference_ids(
    references: Iterable[Any],
) -> Iterable[Tuple[Optional[str], Optional[str]]]:
    """
    Normalize a heterogeneous reference list into (arxiv_id, doi) tuples.

    This is intentionally defensive so it works regardless of how
    IngestedPaperResult.references is structured:

    - If items are strings, we treat them as arxiv_ids.
    - If items are dict-like, we look for 'arxiv_id'/'arxivId' and 'doi',
      and also in a nested 'ids' dict if present.
    - If items are objects, we look for .arxiv_id/.arxivId and .doi.
    """
    for ref in references:
        # Simple string -> arxiv id
        if isinstance(ref, str):
            yield (ref, None)
            continue

        arxiv_id: Optional[str] = None
        doi: Optional[str] = None

        if isinstance(ref, dict):
            arxiv_id = (
                ref.get("arxiv_id")
                or ref.get("arxivId")
                or None
            )
            doi = ref.get("doi") or None

            ids = ref.get("ids")
            if isinstance(ids, dict):
                arxiv_id = arxiv_id or ids.get("arxiv") or ids.get("arxiv_id")
                doi = doi or ids.get("doi") or ids.get("DOI")
        else:
            # Object with attributes (e.g. your Reference dataclass)
            arxiv_id = (
                getattr(ref, "arxiv_id", None)
                or getattr(ref, "arxivId", None)
            )
            doi = getattr(ref, "doi", None)

        if arxiv_id or doi:
            yield (arxiv_id, doi)


# ---------------------------------------------------------------------------
# Single-paper ingest wrapper
# ---------------------------------------------------------------------------

def _ingest_single_seed(
    seed: SeedPaper,
    grobid_client: GrobidClient,
    work_dir: Path,
    use_cache: bool,
) -> Optional[IngestedPaperResult]:
    """
    Ingest a single paper given a SeedPaper and return the IngestedPaperResult.

    For now we support ingestion by arxiv_id; DOI-based ingestion can be
    added later as a separate code path if you add such a helper.
    """
    if not seed.arxiv_id:
        # For now, we cannot ingest purely-DOI seeds without a DOI->PDF pipeline.
        return None

    return ingest_arxiv_paper(
        arxiv_id=seed.arxiv_id,
        work_dir=work_dir,
        grobid_client=grobid_client,
        use_cache=use_cache,
        force_reingest=False,
    )


# ---------------------------------------------------------------------------
# BFS orchestrator
# ---------------------------------------------------------------------------

def bfs_seed_and_expand(
    initial_seeds: Sequence[SeedPaper],
    config: Optional[BFSConfig] = None,
    grobid_client: Optional[GrobidClient] = None,
) -> None:
    """
    Breadth-first ingestion starting from a collection of seed papers.

    For each successfully ingested paper:
      - Its concepts and metadata are added to the graph via
        update_graph_with_ingested_paper.
      - Its reference list is used to enqueue further seeds (up to max_depth).

    De-duplication is done across both arxiv ids and DOIs, so the same
    logical paper will not be processed twice even if it appears under
    different identifiers.

    The updated graph is persisted at the end using save_graph().
    """
    if config is None:
        config = BFSConfig()

    work_dir = config.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    owns_client = grobid_client is None
    if grobid_client is None:
        grobid_client = GrobidClient()

    # Load the latest saved graph, or start a fresh one depending on config
    if config.start_fresh:
        print("[INFO] Starting BFS ingest from a fresh empty graph")
        G: nx.MultiDiGraph = nx.MultiDiGraph()
    else:
        G_loaded = load_latest_graph()
        if G_loaded is None:
            print("[INFO] No existing graph found; starting fresh")
            G = nx.MultiDiGraph()
        else:
            print("[INFO] Loaded existing graph; continuing BFS ingest on top")
            G = G_loaded

    queue: Deque[Tuple[SeedPaper, int]] = deque()
    seen: Set[str] = set()
    ingested_count = 0

    # Initialize queue + seen from initial seeds
    for seed in initial_seeds:
        key = _seen_key(seed.arxiv_id, seed.doi)
        if key is None:
            continue
        if key in seen:
            continue
        seen.add(key)
        queue.append((seed, 0))

    while queue and ingested_count < config.max_papers:
        seed, depth = queue.popleft()
        if depth > config.max_depth:
            continue

        try:
            result = _ingest_single_seed(
                seed=seed,
                grobid_client=grobid_client,
                work_dir=work_dir,
                use_cache=config.use_cache,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to ingest seed {seed}: {exc}")
            continue

        if result is None:
            continue

        # DEBUG: show a bit of what references look like
        if result.references:
            sample = list(result.references)[:5]
            print(
                f"[DEBUG] {seed} -> {len(result.references)} references, sample={sample}"
            )
        else:
            print(f"[DEBUG] {seed} -> no references")

        # Push into the graph (this adds paper, concepts, and citation edges)
        try:
            update_graph_with_ingested_paper(G, result)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to update graph for {seed}: {exc}")
        else:
            ingested_count += 1

        # Enqueue references for next layer
        next_depth = depth + 1
        if next_depth > config.max_depth:
            continue

        for ref_arxiv, ref_doi in _iter_reference_ids(result.references or []):
            key = _seen_key(ref_arxiv, ref_doi)
            if key is None:
                continue
            if key in seen:
                continue
            seen.add(key)
            queue.append((SeedPaper(arxiv_id=ref_arxiv, doi=ref_doi), next_depth))

    # Persist the updated graph
    save_path = save_graph(G)
    print(f"[INFO] Saved updated graph to {save_path}")

    if owns_client:
        try:
            grobid_client.close()
        except Exception:
            pass

    print(
        f"[INFO] BFS ingestion complete. "
        f"Ingested {ingested_count} papers, "
        f"visited {len(seen)} unique ids "
        f"(max_depth={config.max_depth})."
    )
