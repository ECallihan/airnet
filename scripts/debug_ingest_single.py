# scripts/debug_ingest_single.py

from pathlib import Path
from kg_ai_papers.ingest.pipeline import ingest_arxiv_paper
from kg_ai_papers.grobid_client import GrobidClient

import sys


def main(arxiv_id: str) -> None:
    work_dir = Path("data/ingest_debug")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Uses your real signature: config=None, base_url=..., timeout=...
    client = GrobidClient(
        base_url="http://localhost:8070",  # adjust if your GROBID URL is different
        timeout=60,
    )

    result = ingest_arxiv_paper(
        arxiv_id=arxiv_id,
        work_dir=work_dir,
        grobid_client=client,
        use_cache=False,
        force_reingest=True,
    )

    print(f"[debug] Paper: {result.paper.arxiv_id}  title={result.paper.title!r}")
    print(f"[debug] #concept_summaries = {len(result.concept_summaries)}")

    for key, summary in list(result.concept_summaries.items())[:20]:
        print(
            f"  - key={key!r}, base_name={summary.base_name!r}, "
            f"mentions_total={summary.mentions_total}, "
            f"weighted_score={summary.weighted_score}"
        )

    num_refs = len(result.references) if result.references is not None else 0
    print(f"[debug] #references = {num_refs}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.debug_ingest_single <arxiv_id>")
        raise SystemExit(1)
    main(sys.argv[1])
