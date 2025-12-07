# kg_ai_papers/cli/batch_ingest.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence
import pickle

import typer
import networkx as nx
import traceback

from kg_ai_papers.ingest.pipeline import (
    IngestedPaperResult,
    ingest_pdf,  # your single-paper ingestion entrypoint
)
from kg_ai_papers.graph.builder import update_graph_with_ingested_paper
from kg_ai_papers.graph.io import save_graph 

app = typer.Typer(help="AirNet batch ingestion + graph build CLI")


def _load_ids(ids: Sequence[str], ids_file: Optional[Path]) -> List[str]:
    collected: List[str] = []

    if ids_file is not None:
        text = ids_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            collected.append(line)

    collected.extend(ids)

    # De-duplicate while preserving order
    seen = set()
    unique_ids: List[str] = []
    for arxiv_id in collected:
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            unique_ids.append(arxiv_id)

    return unique_ids


def _serialize_ingested(result: IngestedPaperResult) -> dict:
    """
    Best-effort JSON-serializable view of IngestedPaperResult
    for optional debugging/export. Adjust as needed based on your
    actual dataclasses / models.
    """
    paper = result.paper
    if hasattr(paper, "model_dump"):
        paper_dict = paper.model_dump()
    elif hasattr(paper, "dict"):
        paper_dict = paper.dict()
    else:
        paper_dict = getattr(paper, "__dict__", paper)

    concept_summaries = [
        {
            "concept": cs.concept,
            "score": cs.score,
            "section_weight": cs.section_weight,
        }
        for cs in result.concept_summaries
    ]

    return {
        "paper": paper_dict,
        "concept_summaries": concept_summaries,
        "references": list(result.references),
    }


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
) -> None:
    """
    Ingest a batch of arXiv papers and build a citation/concept graph.
    """
    all_ids = _load_ids(ids or [], ids_file)

    if not all_ids:
        typer.echo("ERROR: No arXiv IDs provided (use --id or --ids-file).", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Preparing to ingest {len(all_ids)} papers...")
    work_dir.mkdir(parents=True, exist_ok=True)
    graph_output.parent.mkdir(parents=True, exist_ok=True)

    ingested_results: List[IngestedPaperResult] = []

    # ---- Ingestion loop -----------------------------------------------------
    for idx, arxiv_id in enumerate(all_ids, start=1):
        typer.echo(f"[{idx}/{len(all_ids)}] Ingesting {arxiv_id} ...")
        try:
            result = ingest_pdf(arxiv_id=arxiv_id, work_dir=work_dir)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  !! Failed to ingest {arxiv_id}: {exc!r}", err=True)
            typer.echo("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), err=True)
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

    graph_output.parent.mkdir(parents=True, exist_ok=True)

    # Persist the graph via our dedicated Graph I/O layer.
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
