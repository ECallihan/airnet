# kg_ai_papers/ingest/arxiv_ingest.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import feedparser
import requests

from kg_ai_papers.config.settings import settings
from kg_ai_papers.models.paper import Paper


ARXIV_API_URL = "http://export.arxiv.org/api/query?search_query=id:{id}&max_results=1"


@dataclass
class ArxivMetadata:
    arxiv_id: str          # including version, e.g. "1706.03762v7"
    title: str
    summary: str
    pdf_url: str


def _fetch_arxiv_metadata(arxiv_id: str) -> Optional[ArxivMetadata]:
    """
    Call the arXiv API for a single id and return minimal metadata.

    arxiv_id can be with or without version ("1706.03762" or "1706.03762v7").
    """
    url = ARXIV_API_URL.format(id=arxiv_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    if not feed.entries:
        return None

    entry = feed.entries[0]

    # entry.id is like "http://arxiv.org/abs/1706.03762v7"
    full_id = entry.id.rsplit("/", 1)[-1]  # "1706.03762v7"

    title = entry.title.strip()
    summary = getattr(entry, "summary", "").strip()

    # Find pdf link if present, else construct it
    pdf_url = None
    for link in getattr(entry, "links", []):
        if link.get("type") == "application/pdf":
            pdf_url = link.get("href")
            break

    if pdf_url is None:
        pdf_url = f"https://arxiv.org/pdf/{full_id}.pdf"

    return ArxivMetadata(
        arxiv_id=full_id,
        title=title,
        summary=summary,
        pdf_url=pdf_url,
    )


def _download_pdf(meta: ArxivMetadata, overwrite: bool = False) -> str:
    """
    Download the PDF to data/raw/papers/{arxiv_id}.pdf and return the local path.
    """
    settings.raw_papers_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = settings.raw_papers_dir / f"{meta.arxiv_id}.pdf"

    if pdf_path.exists() and not overwrite:
        return str(pdf_path)

    resp = requests.get(meta.pdf_url, timeout=120)
    resp.raise_for_status()

    with pdf_path.open("wb") as f:
        f.write(resp.content)

    return str(pdf_path)


def ingest_arxiv_ids(
    arxiv_ids: List[str],
    overwrite_pdfs: bool = False,
) -> List[Paper]:
    """
    High-level ingestion function used by the pipeline.

    For each arXiv id:
      - fetch metadata
      - download PDF
      - build a Paper object with arxiv_id (with version), title, abstract, pdf_path

    Returns:
      List[Paper]
    """
    papers: List[Paper] = []

    for raw_id in arxiv_ids:
        meta = _fetch_arxiv_metadata(raw_id)
        if meta is None:
            print(f"[WARN] No arXiv entry found for {raw_id!r}, skipping.")
            continue

        try:
            pdf_path = _download_pdf(meta, overwrite=overwrite_pdfs)
        except Exception as e:
            print(f"[WARN] Failed to download PDF for {meta.arxiv_id}: {e}")
            pdf_path = ""

        paper = Paper(
            arxiv_id=meta.arxiv_id,
            title=meta.title,
            abstract=meta.summary,
            pdf_path=pdf_path,
        )
        papers.append(paper)

    return papers
