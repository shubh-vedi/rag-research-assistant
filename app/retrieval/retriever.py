"""
Research Retriever
===================
Clean pipeline:
    Vector Search (Contextual Embeddings) → Results

Simple because contextual enrichment already happened at ingestion.
No reranker needed.
"""

import logging
from typing import Optional

from langchain_core.documents import Document

from app.config import settings
from app.embedding.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class ResearchRetriever:
    """
    Clean retrieval pipeline:
        1. Vector Search — find candidates by embedding similarity
           (chunks are already contextually enriched, so simple search works great)
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store = vector_store_manager
        logger.info("ResearchRetriever initialized")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User's research question.
            top_k: Number of final documents to return.

        Returns:
            List of relevant documents, ranked by similarity.
        """
        top_k = top_k or settings.retrieval_k  # Use configured k or override

        # Step 1: Vector search (contextually-enriched embeddings)
        results = self.vector_store.similarity_search(query, k=top_k)

        logger.info(f"Retrieved {len(results)} documents for: '{query[:50]}...'")
        return results
