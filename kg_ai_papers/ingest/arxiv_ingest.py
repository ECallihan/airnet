# kg_ai_papers/ingest/arxiv_ingest.py

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import json
import arxiv
import requests

from kg_ai_papers.config.settings import settings
from kg_ai_papers.models.paper import Paper


def _fetch_single_result(arxiv_id: str) -> Optional[arxiv.Result]:
    search = arxiv.Search(
        query=f"id:{arxiv_id}",
        max_results=1,
    )
    for result in search.results():
        return result
    return None


def _download_pdf(pdf_url: str, dest_path: Path, overwrite: bool = False) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not overwrite:
        return dest_path

    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)
    return dest_path


def fetch_papers_and_pdfs(
    arxiv_ids: List[str],
    overwrite_pdfs: bool = False,
) -> List[Paper]:
    """
    Fetch metadata + PDFs for a list of arXiv IDs and return a list of Paper objects.

    - PDFs are stored in settings.raw_papers_dir / {arxiv_id}.pdf
    - Metadata JSON is stored in settings.raw_metadata_dir / {arxiv_id}.json
    """
    papers: List[Paper] = []

    for aid in arxiv_ids:
        result = _fetch_single_result(aid)
        if result is None:
            print(f"[WARN] No arXiv result found for {aid}, skipping.")
            continue

        pdf_path = settings.raw_papers_dir / f"{result.get_short_id()}.pdf"
        print(f"[INGEST] Downloading PDF for {aid} -> {pdf_path}")
        _download_pdf(result.pdf_url, pdf_path, overwrite=overwrite_pdfs)

        paper = Paper(
            arxiv_id=result.get_short_id(),
            title=result.title,
            abstract=result.summary,
            pdf_path=str(pdf_path),
        )

        # Save minimal metadata JSON
        meta_path = settings.raw_metadata_dir / f"{paper.arxiv_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(paper), f, indent=2, ensure_ascii=False)

        papers.append(paper)

    return papers
