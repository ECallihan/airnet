# tests/test_settings.py

from kg_ai_papers.config.settings import get_settings
from kg_ai_papers.config.settings import Settings, RuntimeMode


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

def test_runtime_mode_and_embedding_defaults():
    """
    Sanity check: new capability flags have sensible defaults,
    and do not change behavior unless explicitly overridden.
    """
    settings = Settings()

    # Default mode should be STANDARD so nothing changes
    assert settings.runtime_mode == RuntimeMode.STANDARD

    # Embeddings are enabled by default to preserve existing tests/behavior
    assert settings.enable_embeddings is True

    # Embedding batch size property should map to the STANDARD value by default
    assert settings.embedding_batch_size == settings.embedding_batch_size_standard


def test_runtime_mode_env_override(monkeypatch):
    """
    Ensure runtime_mode can be overridden via environment variable.
    """
    monkeypatch.setenv("AIRNET_RUNTIME_MODE", "light")

    settings = Settings()
    assert settings.runtime_mode == RuntimeMode.LIGHT
