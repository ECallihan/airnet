# tests/test_batch_ingest_cli.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pytest
from typer.testing import CliRunner

from kg_ai_papers.models.paper import Paper
from kg_ai_papers.ingest.pipeline import IngestedPaperResult
from kg_ai_papers.cli.batch_ingest import app as batch_app
import kg_ai_papers.cli.batch_ingest as batch_mod


runner = CliRunner()


def _patch_ingestion(monkeypatch) -> List[Tuple[str, Path]]:
    """
    Patch the ingestion function used by the batch CLI so that tests don't
    hit the real arXiv / GROBID / NLP stack.

    Returns a list that will be populated with (arxiv_id, work_dir) tuples
    in the order they were ingested.
    """
    calls: List[Tuple[str, Path]] = []

    def fake_ingest(*, arxiv_id: str, work_dir: Path, **kwargs):
        # Record call
        calls.append((arxiv_id, work_dir))

        # Minimal but realistic Paper + IngestedPaperResult
        paper = Paper(
            arxiv_id=arxiv_id,
            title=f"Title for {arxiv_id}",
            abstract=f"Abstract for {arxiv_id}",
            pdf_path=str(work_dir / f"{arxiv_id}.pdf"),
        )
        return IngestedPaperResult(
            paper=paper,
            concept_summaries={},
            references=[],
        )

    # Support both older and newer CLI implementations:
    # - newer: uses ingest_arxiv_paper
    # - older: uses ingest_pdf (arxiv_id, work_dir)
    if hasattr(batch_mod, "ingest_arxiv_paper"):
        monkeypatch.setattr(batch_mod, "ingest_arxiv_paper", fake_ingest)
    elif hasattr(batch_mod, "ingest_pdf"):
        # Adapt signature slightly: CLI likely calls ingest_pdf(arxiv_id=..., work_dir=...)
        def fake_ingest_pdf(*, arxiv_id: str, work_dir: Path, **kwargs):
            return fake_ingest(arxiv_id=arxiv_id, work_dir=work_dir, **kwargs)

        monkeypatch.setattr(batch_mod, "ingest_pdf", fake_ingest_pdf)
    else:
        raise RuntimeError(
            "Neither ingest_arxiv_paper nor ingest_pdf found in batch CLI module;"
            " update tests if the CLI has changed."
        )

    return calls


def _patch_graph_layer(monkeypatch):
    """
    Patch update_graph_with_ingested_paper and save_graph to avoid touching
    real graph logic and to observe calls.
    """
    update_calls: List[str] = []
    saved_paths: List[Path] = []

    def fake_update_graph(G, result: IngestedPaperResult):
        # record paper ID used to update graph
        update_calls.append(result.paper.arxiv_id)

    def fake_save_graph(G, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write a tiny marker file so tests can assert it exists
        p.write_bytes(b"FAKE_GRAPH")
        saved_paths.append(p)
        return p

    monkeypatch.setattr(batch_mod, "update_graph_with_ingested_paper", fake_update_graph)
    monkeypatch.setattr(batch_mod, "save_graph", fake_save_graph)

    return update_calls, saved_paths


def test_batch_ingest_single_id(tmp_path: Path, monkeypatch):
    """
    Basic happy-path:
      - single --id
      - ingestion called once
      - graph saved
      - JSON dump created with expected structure.
    """
    ingest_calls = _patch_ingestion(monkeypatch)
    update_calls, saved_paths = _patch_graph_layer(monkeypatch)

    work_dir = tmp_path / "ingest"
    graph_output = tmp_path / "graphs" / "airnet.gpickle"
    json_output = tmp_path / "ingested.json"

    result = runner.invoke(
        batch_app,
        [
            "--id",
            "2401.12345",
            "--work-dir",
            str(work_dir),
            "--graph-output",
            str(graph_output),
            "--dump-ingested-json",
            str(json_output),
        ],
    )

    assert result.exit_code == 0, result.output

    # Ingestion called exactly once with the ID and work_dir we passed
    assert ingest_calls == [("2401.12345", work_dir)]

    # Graph update called once for that same paper
    assert update_calls == ["2401.12345"]

    # Graph was saved to the requested path
    assert saved_paths == [graph_output]
    assert graph_output.exists()
    assert graph_output.read_bytes() == b"FAKE_GRAPH"

    # JSON dump was created with the ingested paper metadata
    assert json_output.exists()
    payload = json_output.read_text(encoding="utf-8")
    data = json.loads(payload)

    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["paper"]["arxiv_id"] == "2401.12345"
    # Minimal sanity check on shape
    assert "concept_summaries" in data[0]
    assert "references" in data[0]


def test_batch_ingest_ids_file_and_cli_merge_and_dedup(tmp_path: Path, monkeypatch):
    """
    Ensure that:
      - IDs from --ids-file and --id are combined.
      - Order is preserved.
      - Duplicates are removed while preserving first occurrence.
    """
    ingest_calls = _patch_ingestion(monkeypatch)
    update_calls, saved_paths = _patch_graph_layer(monkeypatch)

    # ids.txt with duplicates
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("id1\nid2\nid2\n", encoding="utf-8")

    work_dir = tmp_path / "ingest"
    graph_output = tmp_path / "graphs" / "airnet.gpickle"

    result = runner.invoke(
        batch_app,
        [
            "--ids-file",
            str(ids_file),
            "--id",
            "id2",  # duplicate, should be ignored on second/third occurrence
            "--id",
            "id3",
            "--work-dir",
            str(work_dir),
            "--graph-output",
            str(graph_output),
        ],
    )

    assert result.exit_code == 0, result.output

    # The deduped order should be: ["id1", "id2", "id3"]
    ingested_ids = [arxiv_id for (arxiv_id, _wd) in ingest_calls]
    assert ingested_ids == ["id1", "id2", "id3"]

    # update_graph_with_ingested_paper called once per ingested paper
    assert update_calls == ["id1", "id2", "id3"]

    # Graph saved once
    assert saved_paths == [graph_output]
    assert graph_output.exists()


def test_batch_ingest_requires_at_least_one_id(monkeypatch):
    """
    Running the CLI with no --id and no --ids-file should exit with code 1
    and print a helpful error message.
    """
    # Patch ingestion & graph layer just in case, though they should not be used
    _patch_ingestion(monkeypatch)
    _patch_graph_layer(monkeypatch)

    result = runner.invoke(batch_app, [])

    assert result.exit_code != 0
    assert "No arXiv IDs provided" in result.output
