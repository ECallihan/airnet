# kg_ai_papers/grobid_client.py

from __future__ import annotations
import asyncio

import os
from pathlib import Path
from typing import Optional
import httpx
from kg_ai_papers.config.settings import Settings
_settings = Settings()

import requests


class GrobidClientError(Exception):
    """Raised when a GROBID request fails."""


class GrobidClient:
    def __init__(self, base_url: str, max_concurrent: Optional[int] = None) -> None:
        self.base_url = base_url
        self._max_concurrent = max_concurrent or _settings.grobid_max_concurrent_requests
        self._sem = asyncio.Semaphore(self._max_concurrent)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    async def process_fulltext(self, pdf_bytes: bytes) -> str:
        async with self._sem:
            resp = await self._client.post(
                "/api/processFulltextDocument",
                files={"input": ("file.pdf", pdf_bytes, "application/pdf")},
            )
            resp.raise_for_status()
            return resp.text

    @property
    def _fulltext_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/processFulltextDocument"

    def healthcheck(self) -> bool:
        """Return True if GROBID /api/isalive responds with HTTP 200."""
        url = f"{self.base_url.rstrip('/')}/api/isalive"
        try:
            resp = requests.get(url, timeout=5)
        except requests.RequestException:
            return False
        return resp.ok

    def process_pdf(
        self,
        pdf_path: Path | str,
        paper_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Send a PDF to GROBID and save the returned TEI XML.

        Returns the path to the TEI file.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if paper_id is None:
            paper_id = pdf_path.stem

        tei_path = self.tei_dir / f"{paper_id}.tei.xml"

        if tei_path.exists() and not overwrite:
            return tei_path

        files = {"input": (pdf_path.name, pdf_path.open("rb"), "application/pdf")}
        data = {
            "consolidateHeader": 1,
            "consolidateCitations": 1,
        }

        try:
            resp = requests.post(
                self._fulltext_url,
                files=files,
                data=data,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise GrobidClientError(f"Error calling GROBID: {e}") from e

        if not resp.ok:
            raise GrobidClientError(
                f"GROBID returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        tei_path.write_text(resp.text, encoding="utf-8")
        return tei_path
