from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeMode(str, Enum):
    """
    Overall runtime profile for the system.

    LIGHT    - smaller machines / laptops, do cheaper work.
    STANDARD - default behavior (matches existing behavior as closely as possible).
    HEAVY    - big boxes / GPUs, allow more aggressive settings.
    """
    LIGHT = "light"
    STANDARD = "standard"
    HEAVY = "heavy"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        env_prefix="AIRNET_"
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
        description=(
            "Number of top concepts to keep per section when extracting "
            "section-level concepts."
        ),
    )

    PAPER_TOP_CONCEPTS: int = Field(
        default=10,
        description=(
            "Number of top concepts to keep per paper when aggregating "
            "section-level concepts."
        ),
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
        description="Batch size for embedding computation (legacy / upper-case setting).",
    )

    # ------------------------------------------------------------------
    # Runtime / capability knobs
    # ------------------------------------------------------------------
    runtime_mode: RuntimeMode = Field(
        default=RuntimeMode.STANDARD,
        description=(
            "Overall runtime profile: light/standard/heavy. "
            "STANDARD is the default to preserve current behavior."
        ),
    )

    enable_embeddings: bool = Field(
        default=True,
        description=(
            "If False, disable embedding-based operations and use a cheap fallback. "
            "Useful on very constrained machines."
        ),
    )

    embedding_model_name: Optional[str] = Field(
        default=None,
        description=(
            "Optional override for the embedding model name. "
            "If None, SENTENCE_MODEL_NAME is used."
        ),
    )

    embedding_batch_size_light: int = Field(
        default=8,
        description="Batch size hint for LIGHT runtime_mode.",
    )
    embedding_batch_size_standard: int = Field(
        default=32,
        description="Batch size hint for STANDARD runtime_mode.",
    )
    embedding_batch_size_heavy: int = Field(
        default=128,
        description="Batch size hint for HEAVY runtime_mode.",
    )

    grobid_max_concurrent_requests: int = Field(
        default=4,
        description=(
            "Soft limit for concurrent GROBID requests. "
            "Actual enforcement is done in the client layer."
        ),
    )

    max_papers_per_ingest_batch: int = Field(
        default=32,
        description="How many papers to process in one ingest batch.",
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

    @property
    def embedding_batch_size(self) -> int:
        """
        Derived, runtime-mode-aware batch size for embeddings.

        Existing code that uses the old EMBEDDING_BATCH_SIZE constant
        still works. New code can use this property for mode-aware tuning.
        """
        if self.runtime_mode == RuntimeMode.LIGHT:
            return self.embedding_batch_size_light
        if self.runtime_mode == RuntimeMode.HEAVY:
            return self.embedding_batch_size_heavy
        return self.embedding_batch_size_standard


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
