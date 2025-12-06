# kg_ai_papers/config/settings.py

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global project configuration.

    Values can be overridden via environment variables, e.g.:
      - GROBID_URL=http://localhost:8070
      - DATA_DIR=/mnt/data/kg-ai-papers
    or via a .env file in the project root.
    """

    # ------------------------
    # Paths
    # ------------------------
    DATA_DIR: Path = Field(default=Path("data"))
    RAW_PAPERS_DIR: Optional[Path] = None
    RAW_METADATA_DIR: Optional[Path] = None
    PARSED_DIR: Optional[Path] = None
    ENRICHED_DIR: Optional[Path] = None
    GRAPH_DIR: Optional[Path] = None

    # ------------------------
    # External services
    # ------------------------
    GROBID_URL: AnyHttpUrl = Field(default="http://localhost:8070")

    # ------------------------
    # NLP models
    # ------------------------
    SENTENCE_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name for paper/concept embeddings.",
    )
    KEYBERT_MODEL_NAME: Optional[str] = Field(
        default=None,
        description="Model to use inside KeyBERT. If None, KeyBERT uses its default.",
    )

    # ------------------------
    # Concept extraction
    # ------------------------
    MAX_SECTION_CHARS: int = Field(
        default=8000,
        description="Truncate very long sections before keyphrase extraction.",
    )
    PAPER_TOP_CONCEPTS: int = Field(
        default=30,
        description="Default number of concepts to extract per paper.",
    )
    SECTION_TOP_CONCEPTS: int = Field(
        default=10,
        description="Default number of concepts to extract per section.",
    )

    # ------------------------
    # Graph / influence
    # ------------------------
    CITATION_SIM_MIN: float = Field(
        default=0.0,
        description="Minimum cosine similarity for creating a citation influence edge.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="API key for simple header-based auth. If None, auth is disabled.",
    )

    GRAPH_DEFAULT_NAME: str = Field(
        default="graph",
        description="Default graph name to load from data/graph/{name}.gpickle",
    )




    # ------------------------
    # Derived properties
    # ------------------------

    @property
    def raw_papers_dir(self) -> Path:
        return (self.RAW_PAPERS_DIR or (self.DATA_DIR / "raw" / "papers")).resolve()

    @property
    def raw_metadata_dir(self) -> Path:
        return (self.RAW_METADATA_DIR or (self.DATA_DIR / "raw" / "metadata")).resolve()

    @property
    def parsed_dir(self) -> Path:
        return (self.PARSED_DIR or (self.DATA_DIR / "parsed")).resolve()

    @property
    def enriched_dir(self) -> Path:
        return (self.ENRICHED_DIR or (self.DATA_DIR / "enriched")).resolve()

    @property
    def graph_dir(self) -> Path:
        return (self.GRAPH_DIR or (self.DATA_DIR / "graph")).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached accessor so you don't keep re-parsing env/.env.
    """
    settings = Settings()

    # Ensure directories exist
    for d in (
        settings.DATA_DIR,
        settings.raw_papers_dir,
        settings.raw_metadata_dir,
        settings.parsed_dir,
        settings.enriched_dir,
        settings.graph_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    return settings


# convenience
settings = get_settings()
