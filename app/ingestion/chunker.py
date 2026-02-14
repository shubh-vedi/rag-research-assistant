"""
Semantic Document Chunker
Splits documents based on meaning boundaries rather than fixed character counts.
Uses embedding similarity to detect topic shifts for coherent chunk creation.
"""

import logging
import uuid

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)


def semantic_chunk(
    documents: list[Document],
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 90,
) -> list[Document]:
    """
    Split documents using semantic chunking.

    Semantic chunking detects meaning boundaries using embedding similarity
    between consecutive sentences. When the similarity drops significantly
    (indicating a topic shift), it creates a split point. This produces
    coherent thought-unit chunks that improve retrieval accuracy.

    Falls back to RecursiveCharacterTextSplitter if semantic chunking fails.

    Args:
        documents: List of LangChain Documents to chunk.
        breakpoint_threshold_type: Method to detect breakpoints
            ("percentile", "standard_deviation", "interquartile").
        breakpoint_threshold_amount: Threshold value for breakpoint detection.

    Returns:
        List of chunked Document objects with enriched metadata.
    """
    if not documents:
        return []

    logger.info(
        f"Semantic chunking {len(documents)} documents "
        f"(threshold_type={breakpoint_threshold_type}, "
        f"threshold_amount={breakpoint_threshold_amount})"
    )

    try:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            openai_api_key=settings.openai_api_key,
        )

        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

        chunks = chunker.split_documents(documents)

    except Exception as e:
        logger.warning(
            f"Semantic chunking failed: {e}. Falling back to recursive splitter."
        )
        chunks = _fallback_chunk(documents)

    # Enrich metadata for every chunk
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def _fallback_chunk(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Fallback chunking using RecursiveCharacterTextSplitter.

    Used when semantic chunking fails (e.g., API unavailable).

    Args:
        documents: Documents to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    logger.info(
        f"Using fallback chunking (size={chunk_size}, overlap={chunk_overlap})"
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def fixed_size_chunk(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Fixed-size chunking for comparison or lightweight usage.

    Args:
        documents: Documents to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects with enriched metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_length"] = len(chunk.page_content)
        chunk.metadata["chunking_method"] = "fixed_size"

    return chunks
