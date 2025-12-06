# kg_ai_papers/config/settings.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global configuration for the project.

    Uses pydantic-settings to parse from env vars / .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Core paths / services
    # ------------------------------------------------------------------

    DATA_DIR: Path = Field(
        default=Path("data"),
        description="Base data directory for raw/parsed/enriched/graph files.",
    )

    GROBID_URL: str = Field(
        default="http://localhost:8070",
        description="Base URL of the running Grobid service.",
    )

    # ------------------------------------------------------------------
    # NLP / models
    # ------------------------------------------------------------------

    SENTENCE_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name for embeddings.",
    )

    KEYBERT_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name used by KeyBERT for concept extraction.",
    )

    SECTION_TOP_CONCEPTS: int = Field(
        default=5,
        description="Number of top concepts to keep per section when extracting section-level concepts.",
    )

    PAPER_TOP_CONCEPTS: int = Field(
        default=10,
        description="Number of top concepts to keep per paper when aggregating section-level concepts.",
    )

    MAX_SECTION_CHARS: int = Field(
        default=4000,
        description="Maximum number of characters from a section to send into KeyBERT to avoid overly long inputs.",
    )

    # You can add more NLP knobs later if needed, e.g. EMBED_BATCH_SIZE, MIN_CONCEPT_WEIGHT, etc.

    # ------------------------------------------------------------------
    # Graph / API
    # ------------------------------------------------------------------

    GRAPH_DEFAULT_NAME: str = Field(
        default="graph",
        description="Default graph name for data/graph/{name}.gpickle",
    )

    API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="API key for header-based auth. If None, auth is disabled.",
    )

    # ------------------------------------------------------------------
    # Convenience derived paths
    # ------------------------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def raw_papers_dir(self) -> Path:
        return self.raw_dir / "papers"

    @property
    def raw_metadata_dir(self) -> Path:
        return self.raw_dir / "metadata"

    @property
    def parsed_dir(self) -> Path:
        return self.DATA_DIR / "parsed"

    @property
    def enriched_dir(self) -> Path:
        return self.DATA_DIR / "enriched"

    @property
    def graph_dir(self) -> Path:
        return self.DATA_DIR / "graph"


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _settings.raw_papers_dir.mkdir(parents=True, exist_ok=True)
        _settings.raw_metadata_dir.mkdir(parents=True, exist_ok=True)
        _settings.parsed_dir.mkdir(parents=True, exist_ok=True)
        _settings.enriched_dir.mkdir(parents=True, exist_ok=True)
        _settings.graph_dir.mkdir(parents=True, exist_ok=True)
    return _settings


settings = get_settings()
