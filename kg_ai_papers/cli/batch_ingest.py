# kg_ai_papers/cli/batch_ingest.py

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import traceback
import typer

from kg_ai_papers.ingest.pipeline import (
    IngestedPaperResult,
    ingest_arxiv_paper,  # high-level arXiv entrypoint
)
from kg_ai_papers.graph.builder import update_graph_with_ingested_paper
from kg_ai_papers.graph.io import save_graph


app = typer.Typer(help="AirNet batch ingestion + graph build CLI")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ids(ids: Sequence[str], ids_file: Optional[Path]) -> List[str]:
    """Combine IDs from CLI options and an optional file, preserving order."""
    collected: List[str] = []

    if ids_file is not None:
        text = ids_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line:
                collected.append(line)

    for arxiv_id in ids:
        arxiv_id = arxiv_id.strip()
        if arxiv_id:
            collected.append(arxiv_id)

    # De-duplicate while preserving order
    seen = set()
    unique_ids: List[str] = []
    for arxiv_id in collected:
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            unique_ids.append(arxiv_id)

    return unique_ids


def _paper_to_dict(paper: Any) -> Dict[str, Any]:
    """Best-effort conversion of a Paper-like object to a plain dict."""
    if paper is None:
        return {}

    if hasattr(paper, "model_dump"):
        # Pydantic-style
        return paper.model_dump()

    if is_dataclass(paper):
        return asdict(paper)

    if hasattr(paper, "__dict__"):
        return {
            k: v
            for k, v in vars(paper).items()
            if not k.startswith("_")
        }

    return {"value": str(paper)}


def _serialize_ingested(result: Any) -> dict:
    """
    JSON-serialisable view of an ingestion result.

    Handles both:
      - IngestedPaperResult (preferred, has `.paper`)
      - IngestedPaper-style objects (fallback – no `.paper`)
    """
    # Figure out the "paper object"
    if hasattr(result, "paper"):
        paper_obj = result.paper
    else:
        paper_obj = result

    paper_dict = _paper_to_dict(paper_obj)

    # Concept summaries may be a dict[str, ConceptSummary] or other structure
    raw_cs = getattr(result, "concept_summaries", {}) or {}
    concept_summaries_list: List[Dict[str, Any]] = []

    if isinstance(raw_cs, dict):
        for key, cs in raw_cs.items():
            concept_summaries_list.append(
                {
                    "concept": key,
                    "score": getattr(cs, "score", None),
                    "section_weight": getattr(cs, "section_weight", None),
                }
            )
    else:
        # Fallback: treat as iterable of ConceptSummary-like objects
        for cs in raw_cs:
            concept_summaries_list.append(
                {
                    "concept": getattr(cs, "concept", None),
                    "score": getattr(cs, "score", None),
                    "section_weight": getattr(cs, "section_weight", None),
                }
            )

    # References: prefer result.references, else try paper.references
    refs = getattr(result, "references", None)
    if refs is None:
        refs = getattr(paper_obj, "references", []) or []

    simple_refs: List[str] = []
    for r in refs:
        if isinstance(r, str):
            simple_refs.append(r)
        else:
            # Try to pull an arxiv_id attribute out of a Reference-like object
            arxiv_id = getattr(r, "arxiv_id", None)
            if arxiv_id is not None:
                simple_refs.append(arxiv_id)

    return {
        "paper": paper_dict,
        "concept_summaries": concept_summaries_list,
        "references": simple_refs,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@app.command()
def main(
    ids: List[str] = typer.Option(
        None,
        "--id",
        "-i",
        help="ArXiv ID to ingest. Can be passed multiple times.",
    ),
    ids_file: Optional[Path] = typer.Option(
        None,
        "--ids-file",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a text file with one arXiv ID per line.",
    ),
    work_dir: Path = typer.Option(
        Path("data/ingest"),
        "--work-dir",
        "-w",
        help="Directory for PDFs / TEI files used during ingestion.",
    ),
    graph_output: Path = typer.Option(
        Path("data/graphs/airnet.gpickle"),
        "--graph-output",
        "-g",
        help="Path to write the resulting NetworkX graph (gpickle).",
    ),
    dump_ingested_json: Optional[Path] = typer.Option(
        None,
        "--dump-ingested-json",
        help=(
            "If set, dump a JSON list of ingested paper results "
            "for debugging / inspection."
        ),
    ),
    use_cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Enable or disable ingestion cache (PDF/TEI/concepts).",
    ),
    force_reingest: bool = typer.Option(
        False,
        "--force-reingest",
        help="Ignore cached ingestion results and recompute for all papers.",
    ),
) -> None:
    """
    Ingest a batch of arXiv papers and build a citation/concept graph.
    """
    all_ids = _load_ids(ids or [], ids_file)

    if not all_ids:
        typer.echo("ERROR: No arXiv IDs provided (use --id or --ids-file).", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Preparing to ingest {len(all_ids)} papers...")

    work_dir = work_dir  # keep as Path
    work_dir.mkdir(parents=True, exist_ok=True)
    graph_output.parent.mkdir(parents=True, exist_ok=True)

    ingested_results: List[IngestedPaperResult] = []

    # ---- Ingestion loop -----------------------------------------------------
    for idx, arxiv_id in enumerate(all_ids, start=1):
        typer.echo(f"[{idx}/{len(all_ids)}] Ingesting {arxiv_id} ...")
        try:
            # High-level entrypoint designed for this use-case
            result = ingest_arxiv_paper(
                arxiv_id=arxiv_id,
                work_dir=work_dir,
                use_cache=use_cache,
                force_reingest=force_reingest,
            )
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  !! Failed to ingest {arxiv_id}: {exc!r}", err=True)
            typer.echo(
                "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
                err=True,
            )
            continue

        ingested_results.append(result)

    if not ingested_results:
        typer.echo("No papers were successfully ingested. Exiting.", err=True)
        raise typer.Exit(code=1)

    # ---- Build NetworkX graph ----------------------------------------------
    typer.echo("Building NetworkX graph from ingested papers...")
    G = nx.MultiDiGraph()

    for result in ingested_results:
        update_graph_with_ingested_paper(G, result)

    typer.echo(
        f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )

    out_path = save_graph(G, graph_output)
    typer.echo(f"Graph written to {out_path}")

    # ---- Optional JSON dump of ingestion outputs ---------------------------
    if dump_ingested_json is not None:
        typer.echo(f"Dumping ingested metadata to {dump_ingested_json}")
        payload = [_serialize_ingested(r) for r in ingested_results]
        dump_ingested_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    app()
