# kg_ai_papers/config/settings.py

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeMode(str, Enum):
    LIGHT = "light"        # laptop / small VM
    STANDARD = "standard"  # default behavior (current behavior)
    HEAVY = "heavy"        # big box / GPU, allow more aggressive settings


class Settings(BaseSettings):
    """
    Global configuration for the project.

    Uses pydantic-settings to parse from environment variables and .env.
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
        description=(
            "SentenceTransformer model name used by KeyBERT for concept extraction. "
            "Usually the same as SENTENCE_MODEL_NAME."
        ),
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
        description=(
            "Maximum number of characters from a section to send into KeyBERT "
            "to avoid overly long inputs."
        ),
    )

    EMBEDDING_DEVICE: str = Field(
        default="auto",
        description="Device for embedding model: 'auto', 'cpu', or 'cuda'.",
    )

    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        description="Batch size for embedding computation.",
    )

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

    runtime_mode: RuntimeMode = Field(
        RuntimeMode.STANDARD,
        description="Overall runtime profile: light/standard/heavy. "
                    "Default is STANDARD to preserve existing behavior.",
    )

    # NLP / embeddings
    enable_embeddings: bool = Field(
        True,
        description="If False, disable embedding-based operations. "
                    "Useful on very constrained machines.",
    )
    embedding_model_name: Optional[str] = Field(
        None,
        description="Override embedding model name. If None, use the existing default "
                    "model configured in nlp/embedding.py.",
    )
    embedding_batch_size_light: int = 8
    embedding_batch_size_standard: int = 32
    embedding_batch_size_heavy: int = 128

    # GROBID / IO
    grobid_max_concurrent_requests: int = Field(
        4,
        description="Soft limit for concurrent GROBID requests. "
                    "Actual enforcement is done in the client layer.",
    )

    max_papers_per_ingest_batch: int = Field(
        32,
        description="How many papers to process in one batch in bulk pipelines.",
    )

    class Config:
        env_prefix = "AIRNET_"
        env_file = ".env"


    # ------------------------------------------------------------------
    # Convenience derived paths
    # ------------------------------------------------------------------
    @property
    def embedding_batch_size(self) -> int:
        if self.runtime_mode == RuntimeMode.LIGHT:
            return self.embedding_batch_size_light
        if self.runtime_mode == RuntimeMode.HEAVY:
            return self.embedding_batch_size_heavy
        return self.embedding_batch_size_standard

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
    """
    Singleton-style accessor so we only construct Settings once and
    ensure directories exist on first access.
    """
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
