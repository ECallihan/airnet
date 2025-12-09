"""
Batch-ingest arXiv papers into the AirNet knowledge graph.

Usage examples:

    # Basic: read IDs from a text file, one per line
    python -m kg_ai_papers.cli.batch_ingest_arxiv \
        --ids-file data/arxiv_ids.txt \
        --work-dir data/ingest \
        --tag demo-run-01

    # Or a quick one-off from the shell
    python -m kg_ai_papers.cli.batch_ingest_arxiv \
        --ids 2401.00001,2401.00002,2401.00003 \
        --work-dir data/ingest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Set

import networkx as nx

from kg_ai_papers.graph.storage import load_latest_graph, save_graph
from kg_ai_papers.graph.builder import update_graph_with_ingested_paper
from kg_ai_papers.ingest.pipeline import ingest_arxiv_paper, IngestedPaperResult


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-ingest arXiv papers into the AirNet graph.",
    )
    parser.add_argument(
        "--ids",
        help="Comma-separated list of arXiv IDs (e.g. 2401.00001,2401.00002).",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Path to a text file with one arXiv ID per line.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Directory used for intermediate ingestion artifacts (PDFs, TEI, cache).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Reuse cached PDFs/TEI if available (default: True).",
    )
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Disable ingestion cache and always refetch/reprocess.",
    )
    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Force re-ingestion even if cached artifacts exist.",
    )
    parser.add_argument(
        "--start-from-empty",
        action="store_true",
        help="Start from an empty in-memory graph instead of loading the latest saved one.",
    )
    parser.add_argument(
        "--tag",
        help=(
            "Tag/name for the saved graph snapshot. "
            "If omitted, a timestamp-based tag is used."
        ),
    )
    return parser.parse_args(argv)


def _collect_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    seen: Set[str] = set()

    # From --ids
    if args.ids:
        for raw in args.ids.split(","):
            s = raw.strip()
            if s and s not in seen:
                seen.add(s)
                ids.append(s)

    # From --ids-file
    if args.ids_file:
        with args.ids_file.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and s not in seen:
                    seen.add(s)
                    ids.append(s)

    if not ids:
        raise SystemExit("No arXiv IDs provided. Use --ids and/or --ids-file.")

    return ids


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_args(argv)
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    ids = _collect_ids(args)
    print(f"[batch-ingest] Will ingest {len(ids)} paper(s): {', '.join(ids)}")

    # Load or initialize the graph
    if args.start_from_empty:
        G = nx.MultiDiGraph()
        print("[batch-ingest] Starting from an empty graph.")
    else:
        try:
            G = load_latest_graph()
            print(
                f"[batch-ingest] Loaded existing graph with "
                f"{G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
            )
        except Exception:
            print("[batch-ingest] No existing graph found; starting from empty.")
            G = nx.MultiDiGraph()

    succeeded: List[str] = []
    failed: List[str] = []

    for arxiv_id in ids:
        print(f"[batch-ingest] Ingesting {arxiv_id!r} ...", end="", flush=True)
        try:
            result: IngestedPaperResult = ingest_arxiv_paper(
                arxiv_id=arxiv_id,
                work_dir=work_dir,
                use_cache=args.use_cache,
                force_reingest=args.force_reingest,
            )
            update_graph_with_ingested_paper(G, result)
        except Exception as exc:  # pragma: no cover - CLI/debug path
            print(f" FAILED ({exc})")
            failed.append(arxiv_id)
            continue

        print(" ok")
        succeeded.append(arxiv_id)

    # Save updated graph if anything succeeded
    if succeeded:
        tag = args.tag or time.strftime("batch-%Y%m%d-%H%M%S")
        print(
            f"[batch-ingest] Succeeded for {len(succeeded)} paper(s), "
            f"{len(failed)} failed. Saving graph snapshot with tag {tag!r}..."
        )
        save_graph(G, tag)
        print(
            f"[batch-ingest] Final graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges."
        )
    else:
        print(
            "[batch-ingest] No papers ingested successfully; "
            "graph will not be saved."
        )

    if failed:
        print("[batch-ingest] Failed IDs:", ", ".join(failed))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
