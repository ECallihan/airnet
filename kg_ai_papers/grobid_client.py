# kg_ai_papers/grobid_client.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests


class GrobidClientError(Exception):
    """Raised when a GROBID request fails."""


class GrobidClient:
    """
    Minimal client for GROBID's processFulltextDocument endpoint.

    Usage:
        client = GrobidClient()
        tei_path = client.process_pdf(pdf_path, paper_id="arxiv-2401.00001")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        tei_dir: Optional[Path | str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url or os.getenv("GROBID_URL", "http://localhost:8070")
        self.timeout = timeout

        if tei_dir is None:
            tei_dir = os.getenv("TEI_OUTPUT_DIR", "data/tei")
        self.tei_dir = Path(tei_dir)
        self.tei_dir.mkdir(parents=True, exist_ok=True)

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
