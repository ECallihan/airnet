# kg_ai_papers/cli.py

from __future__ import annotations

from pathlib import Path
from typing import List

import typer

from kg_ai_papers.grobid_client import GrobidClient
from kg_ai_papers.ingest.bfs_seed_expand import (
    SeedPaper,
    BFSConfig,
    bfs_seed_and_expand,
)

app = typer.Typer(help="AirNet command-line tools")


def _parse_seed_arg(raw: str) -> SeedPaper:
    """
    Heuristic parsing for seed IDs:

    - If it starts with '10.' -> treat as DOI
    - Otherwise, treat as arXiv ID
      (works well for modern IDs like '2401.12345')
    """
    s = raw.strip()
    if not s:
        raise typer.BadParameter("Seed id cannot be empty")

    if s.startswith("10."):
        # DOI
        return SeedPaper(arxiv_id=None, doi=s)

    # Default: arXiv ID
    return SeedPaper(arxiv_id=s)


@app.command("bfs-ingest")
def bfs_ingest(
    ids: List[str] = typer.Argument(
        ...,
        help="Seed arXiv IDs or DOIs to start BFS from (e.g. 2401.00001 10.1000/xyz123 ...).",
    ),
    max_papers: int = typer.Option(
        50,
        "--max-papers",
        "-n",
        help="Maximum total number of papers to ingest (including seeds).",
    ),
    max_depth: int = typer.Option(
        1,
        "--max-depth",
        "-d",
        help="Maximum BFS depth from seeds (0 = just seeds, 1 = direct references, etc.).",
    ),
    use_cache: bool = typer.Option(
        True,
        "--use-cache/--no-cache",
        help="Reuse ingest cache for papers already processed.",
    ),
    work_dir: Path = typer.Option(
        Path("data/bfs_ingest"),
        "--work-dir",
        "-w",
        help="Working directory for PDFs/TEI/cache for this BFS run.",
    ),
) -> None:
    """
    Breadth-first ingest starting from the given seed IDs.

    Example:
        python -m kg_ai_papers.cli bfs-ingest 2401.00001 2401.01234 -n 100 -d 2
    """
    seeds: List[SeedPaper] = [_parse_seed_arg(x) for x in ids]

    cfg = BFSConfig(
        max_papers=max_papers,
        max_depth=max_depth,
        use_cache=use_cache,
        work_dir=work_dir,
    )

    client = GrobidClient()

    bfs_seed_and_expand(
        initial_seeds=seeds,
        config=cfg,
        grobid_client=client,
    )


if __name__ == "__main__":
    app()
