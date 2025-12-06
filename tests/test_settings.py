# tests/test_settings.py

from kg_ai_papers.config.settings import get_settings


def test_settings_paths_exist(tmp_path, monkeypatch):
    # Override DATA_DIR for test
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    settings = get_settings()

    assert settings.DATA_DIR.exists()
    assert settings.raw_papers_dir.exists()
    assert settings.raw_metadata_dir.exists()
    assert settings.parsed_dir.exists()
    assert settings.enriched_dir.exists()
    assert settings.graph_dir.exists()
