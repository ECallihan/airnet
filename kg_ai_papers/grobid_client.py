# kg_ai_papers/grobid_client.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any
import os

import requests

from kg_ai_papers.config.settings import settings


@dataclass
class GrobidClientConfig:
    """
    Configuration for talking to a GROBID server.
    """

    base_url: str
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "GrobidClientConfig":
        """
        Build configuration from environment variables and global settings.

        Priority for base_url:
        - AIRNET_GROBID_URL
        - GROBID_URL
        - settings.GROBID_URL
        """
        base_url = (
            os.getenv("AIRNET_GROBID_URL")
            or os.getenv("GROBID_URL")
            or settings.GROBID_URL
        )
        base_url = base_url.rstrip("/")

        timeout_str = os.getenv("AIRNET_GROBID_TIMEOUT") or os.getenv("GROBID_TIMEOUT")
        timeout = int(timeout_str) if timeout_str is not None else 60

        return cls(base_url=base_url, timeout=timeout)


class GrobidClientError(RuntimeError):
    """
    Error raised when a GROBID request fails.

    Tests import this as:
        from kg_ai_papers.grobid_client import GrobidClientError
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        url: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url


class GrobidClient:
    """
    Minimal HTTP client for GROBID, matching the test expectations.

    Methods:
        - healthcheck() -> bool
        - is_alive() -> bool (alias)
        - process_pdf(pdf_path, paper_id=None) -> Path
        - parse_pdf_to_tei(pdf_path, paper_id=None) -> Path (alias)
    """

    def __init__(
        self,
        config: Optional[GrobidClientConfig] = None,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        if config is None:
            config = GrobidClientConfig.from_env()

        if base_url is not None:
            config.base_url = base_url.rstrip("/")
        if timeout is not None:
            config.timeout = timeout

        self.config = config
        self.base_url: str = config.base_url
        self.timeout: int = config.timeout

    # ------------------------------------------------------------------
    # Healthcheck
    # ------------------------------------------------------------------
    def is_alive(self) -> bool:
        """
        Call /api/isalive and return True if HTTP 200 and body contains 'true'.
        """
        url = f"{self.base_url}/api/isalive"
        try:
            resp = requests.get(url, timeout=self.timeout)
        except Exception:
            return False

        if not resp.ok:
            return False

        text = (resp.text or "").strip().lower()
        return "true" in text

    def healthcheck(self) -> bool:
        """
        Public healthcheck method used by tests.
        """
        return self.is_alive()

    # ------------------------------------------------------------------
    # Core PDF -> TEI call
    # ------------------------------------------------------------------
    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        paper_id: Optional[str] = None,
    ) -> Path:
        """
        Send a PDF to GROBID /api/processFulltextDocument and return the TEI path.

        If paper_id is provided, it is used for the TEI filename stem;
        otherwise we derive it from pdf_path.stem.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        url = f"{self.base_url}/api/processFulltextDocument"

        files = {"input": pdf_path.open("rb")}
        data: Dict[str, Any] = {
            "consolidateHeader": 1,
            "consolidateCitations": 0,
            "teiCoordinates": "none",
        }

        try:
            resp = requests.post(url, files=files, data=data, timeout=self.timeout)
        except Exception as exc:
            raise GrobidClientError(
                f"Error connecting to GROBID at {url}: {exc}",
                url=url,
            ) from exc
        finally:
            files["input"].close()

        if not resp.ok:
            raise GrobidClientError(
                f"GROBID returned HTTP {resp.status_code} for {url}",
                status_code=resp.status_code,
                url=url,
            )

        # Determine TEI output path
        stem = paper_id if paper_id is not None else pdf_path.stem
        tei_dir = settings.DATA_DIR / "tei"
        tei_dir.mkdir(parents=True, exist_ok=True)
        tei_path = tei_dir / f"{stem}.tei.xml"

        tei_path.write_text(resp.text, encoding="utf-8")
        return tei_path

    def parse_pdf_to_tei(
        self,
        pdf_path: Union[str, Path],
        paper_id: Optional[str] = None,
    ) -> Path:
        """
        Backwards-compatible alias for process_pdf, used by some pipeline code.
        """
        return self.process_pdf(pdf_path, paper_id=paper_id)
