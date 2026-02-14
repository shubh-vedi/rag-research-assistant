"""
Embedding Model Wrapper
Provides a unified interface for creating embedding models.
"""

import logging

from langchain_openai import OpenAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


def get_embedding_model(
    model: str | None = None,
    dimensions: int | None = None,
) -> OpenAIEmbeddings:
    """
    Create and return an OpenAI embedding model instance.

    Uses text-embedding-3-small by default for cost efficiency.
    The model supports dimensionality reduction (e.g., 512 instead of 1536)
    for faster similarity search with minimal quality loss.

    For multilingual workloads, consider BAAI/bge-m3 instead.

    Args:
        model: Embedding model name (defaults to settings.embedding_model).
        dimensions: Embedding dimensions (defaults to settings.embedding_dimensions).

    Returns:
        Configured OpenAIEmbeddings instance.
    """
    model = model or settings.embedding_model
    dimensions = dimensions or settings.embedding_dimensions

    logger.info(f"Initializing embedding model: {model} (dims={dimensions})")

    return OpenAIEmbeddings(
        model=model,
        dimensions=dimensions,
        openai_api_key=settings.openai_api_key,
    )
