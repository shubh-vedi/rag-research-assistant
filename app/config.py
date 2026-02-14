"""
Application configuration using Pydantic Settings.
Loads environment variables from .env file with validation and defaults.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env with override=True so project .env takes precedence
# over stale shell environment variables
load_dotenv(override=True)


class Settings(BaseSettings):
    """Central configuration for the RAG Research Assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────
    openai_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""

    # ── ChromaDB ──────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "research_docs"

    # ── Pinecone (optional) ───────────────────────
    pinecone_index_name: str = "research-assistant"

    # ── Database ──────────────────────────────────
    database_url: str = "sqlite:///./data/sample.db"

    # ── Application ───────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"
    upload_dir: str = "./data/uploads"

    # ── LLM Settings ─────────────────────────────
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 512

    # ── Retrieval Settings ────────────────────────
    semantic_weight: float = 0.6
    bm25_weight: float = 0.4
    retrieval_k: int = 10

    def ensure_directories(self) -> None:
        """Create required data directories if they don't exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = Settings()
