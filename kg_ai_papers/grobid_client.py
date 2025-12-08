from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import requests

from kg_ai_papers.config.settings import settings


# ---------------------------------------------------------------------------
# Thread pool for parallel GROBID calls (used by parse_batch)
# ---------------------------------------------------------------------------

def _default_max_workers() -> int:
    try:
        return max(1, cpu_count() // 2)
    except NotImplementedError:
        return 1


_GROBID_MAX_WORKERS: int = getattr(
    settings,
    "grobid_max_workers",
    _default_max_workers(),
)

_executor = ThreadPoolExecutor(max_workers=_GROBID_MAX_WORKERS)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GrobidClient:
    """
    Lightweight client for a running GROBID service.

    Typical usage:

        client = GrobidClient()  # uses settings.grobid_url
        tei_path = client.parse_pdf_to_tei(Path("paper.pdf"))

    The main entry point for the ingestion pipeline is `parse_pdf_to_tei`,
    which takes a local PDF path and returns the path of the TEI XML file.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 60,
        session: Optional[requests.Session] = None,
    ) -> None:
        # Use Settings.GROBID_URL as the default
        raw_url = base_url or settings.GROBID_URL
        self.base_url: str = raw_url.rstrip("/")
        self.timeout: int = timeout
        self.session: requests.Session = session or requests.Session()

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def is_alive(self) -> bool:
        """
        Return True if the remote GROBID service responds on /api/isalive.
        """
        url = f"{self.base_url}/api/isalive"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def parse_pdf_to_tei(
        self,
        pdf_path: Path,
        tei_path: Optional[Path] = None,
        consolidate_header: int = 1,
        consolidate_citations: int = 1,
        include_raw_citations: int = 0,
    ) -> Path:
        """
        Send a single PDF to GROBID's processFulltextDocument endpoint and
        write the TEI XML response to disk.

        Parameters
        ----------
        pdf_path:
            Local path to the input PDF.
        tei_path:
            Optional explicit output path for the TEI XML. If omitted, we
            use `pdf_path.with_suffix(".tei.xml")`.
        consolidate_header:
            GROBID consolidateHeader parameter (0/1).
        consolidate_citations:
            GROBID consolidateCitations parameter (0/1).
        include_raw_citations:
            GROBID includeRawCitations parameter (0/1).

        Returns
        -------
        Path to the TEI XML file on disk.
        """
        pdf_path = Path(pdf_path)
        if tei_path is None:
            tei_path = pdf_path.with_suffix(".tei.xml")
        else:
            tei_path = Path(tei_path)

        url = f"{self.base_url}/api/processFulltextDocument"

        with pdf_path.open("rb") as f:
            files = {"input": ("file.pdf", f, "application/pdf")}
            data = {
                "consolidateHeader": str(consolidate_header),
                "consolidateCitations": str(consolidate_citations),
                "includeRawCitations": str(include_raw_citations),
            }

            resp = self.session.post(
                url,
                files=files,
                data=data,
                timeout=self.timeout,
            )

        # Raise for non-2xx, so ingestion can log and continue
        resp.raise_for_status()

        tei_path.write_bytes(resp.content)
        return tei_path

    def parse_batch(
        self,
        pdf_paths: Sequence[Path],
        consolidate_header: int = 1,
        consolidate_citations: int = 1,
        include_raw_citations: int = 0,
    ) -> List[Path]:
        """
        Convenience helper: process many PDFs in parallel using the shared
        thread pool configured by settings.grobid_max_workers.

        Returns a list of TEI paths in the same order as pdf_paths.
        """
        pdf_paths = [Path(p) for p in pdf_paths]

        futures = [
            _executor.submit(
                self.parse_pdf_to_tei,
                pdf_path=p,
                tei_path=None,
                consolidate_header=consolidate_header,
                consolidate_citations=consolidate_citations,
                include_raw_citations=include_raw_citations,
            )
            for p in pdf_paths
        ]

        results: List[Path] = []
        for f in futures:
            # Any exception here will propagate to the caller, which is fine:
            # ingestion code can catch/log and move on.
            results.append(f.result())

        return results

    # ------------------------------------------------------------------
    # Context-manager helpers (optional sugar)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close the underlying requests.Session.
        """
        try:
            self.session.close()
        except Exception:
            # Not critical, swallow errors on shutdown.
            pass

    def __enter__(self) -> "GrobidClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
