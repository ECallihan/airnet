# kg_ai_papers/parsing/grobid_client.py

from __future__ import annotations

import os
from typing import Dict, Any

import requests

from kg_ai_papers.config.settings import settings


class GrobidClientError(Exception):
    """
    Domain-specific error for anything that goes wrong talking to Grobid.
    """


def process_fulltext(pdf_path: str) -> str:
    """
    Send a PDF to Grobid and return TEI XML as string.

    This function is intentionally strict about HTTP status codes,
    but it wraps *all* request-related errors into GrobidClientError
    so the pipeline can catch them and continue gracefully.
    """
    url = settings.GROBID_URL.rstrip("/") + "/api/processFulltextDocument"

    params: Dict[str, Any] = {
        "consolidateHeader": 1,
        "consolidateCitations": 1,
        "includeRawCitations": 1,
        "includeRawAffiliations": 0,
    }

    try:
        with open(pdf_path, "rb") as f:
            files = {
                "input": (os.path.basename(pdf_path), f, "application/pdf"),
            }

            response = requests.post(
                url,
                files=files,
                data=params,
                timeout=120,
            )
    except requests.exceptions.RequestException as e:
        # Connection errors, timeouts, DNS, etc.
        raise GrobidClientError(
            f"Error contacting Grobid at {url}: {e}"
        ) from e

    if response.status_code != 200:
        # Grobid responded but with an error
        raise GrobidClientError(
            f"Grobid error {response.status_code}: {response.text[:200]}"
        )

    return response.text
