from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


@dataclass
class GrobidClientConfig:
    """
    Configuration for talking to a GROBID server.

    base_url:
        Root URL of the GROBID service, e.g. "http://localhost:8070".
    timeout:
        Request timeout in seconds.
    pdf_dir / tei_dir:
        Default locations for PDFs and TEI output. These are mostly here so
        they exist as attributes that other parts of the code can rely on.
    """

    base_url: str = os.getenv("GROBID_URL", "http://localhost:8070")
    timeout: int = 60
    pdf_dir: Path = Path("data/grobid/pdf")
    tei_dir: Path = Path("data/grobid/tei")


class GrobidClient:
    """
    Very small HTTP client for GROBID, just enough for our pipeline:

      - is_alive() → bool
      - process_pdf(pdf_path: Path) -> Path (path to TEI XML)

    It uses /api/isalive and /api/processFulltextDocument.
    """

    def __init__(
        self,
        config: Optional[GrobidClientConfig] = None,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        pdf_dir: Optional[Path] = None,
        tei_dir: Optional[Path] = None,
    ) -> None:
        if config is None:
            config = GrobidClientConfig()

        # Allow direct kwargs to override config defaults
        if base_url is not None:
            config.base_url = base_url
        if timeout is not None:
            config.timeout = timeout
        if pdf_dir is not None:
            config.pdf_dir = pdf_dir
        if tei_dir is not None:
            config.tei_dir = tei_dir

        self.config = config

        # Expose pdf_dir / tei_dir as attributes for callers that expect them.
        self.pdf_dir: Path = config.pdf_dir
        self.tei_dir: Path = config.tei_dir

        # Ensure directories exist
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.tei_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Health check
    # ------------------------------------------------------------------ #

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    @property
    def timeout(self) -> int:
        return self.config.timeout

    def is_alive(self) -> bool:
        """
        Check if the GROBID service reports itself as alive.
        """
        url = f"{self.base_url}/api/isalive"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        text = resp.text.strip().lower()
        return "true" in text
