# kg_ai_papers/cli/bfs_ingest.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

from kg_ai_papers.ingest.bfs_seed_expand import (
    SeedPaper,
    BFSConfig,
    bfs_seed_and_expand,
)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BFS-style ingestion starting from seed arXiv IDs.",
    )
    parser.add_argument(
        "--seed-ids",
        help="Comma-separated list of seed arXiv IDs (e.g. 2401.00001,2401.00002).",
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        help="Text file with one seed arXiv ID per line.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=200,
        help="Maximum number of papers to ingest (default: 200).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="BFS depth: 0 = seeds only, 1 = their references, etc. (default: 2).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/bfs_ingest"),
        help="Directory for PDFs/TEI/cache (default: data/bfs_ingest).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Reuse ingestion cache if available (default: True).",
    )
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Disable ingestion cache.",
    )
    return parser.parse_args(argv)


def _collect_seed_ids(args: argparse.Namespace) -> List[str]:
    seeds: List[str] = []
    seen: set[str] = set()

    if args.seed_ids:
        for raw in args.seed_ids.split(","):
            s = raw.strip()
            if s and s not in seen:
                seen.add(s)
                seeds.append(s)

    if args.seed_file:
        with args.seed_file.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and s not in seen:
                    seen.add(s)
                    seeds.append(s)

    if not seeds:
        raise SystemExit("No seeds provided. Use --seed-ids and/or --seed-file.")

    return seeds


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_args(argv)
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    seed_ids = _collect_seed_ids(args)
    print(f"[bfs-ingest] Seeds: {', '.join(seed_ids)}")

    seeds = [SeedPaper(arxiv_id=s) for s in seed_ids]

    config = BFSConfig(
        max_papers=args.max_papers,
        max_depth=args.max_depth,
        use_cache=args.use_cache,
        work_dir=work_dir,
    )

    bfs_seed_and_expand(
        initial_seeds=seeds,
        config=config,
        grobid_client=None,  # let the helper construct/own it
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
