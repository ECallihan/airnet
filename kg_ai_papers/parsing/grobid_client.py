# kg_ai_papers/parsing/grobid_client.py

import os
import requests

from kg_ai_papers.config.settings import settings


class GrobidClientError(Exception):
    pass


def process_fulltext(pdf_path: str) -> str:
    """
    Send a PDF to Grobid and return TEI XML as string.
    """
    url = settings.GROBID_URL.rstrip("/") + "/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
        params = {
            "consolidateHeader": 1,
            "consolidateCitations": 1,
            "includeRawCitations": 1,
            "includeRawAffiliations": 0,
        }
        response = requests.post(url, files=files, data=params, timeout=120)
    if response.status_code != 200:
        raise GrobidClientError(
            f"Grobid error {response.status_code}: {response.text[:200]}"
        )
    return response.text
