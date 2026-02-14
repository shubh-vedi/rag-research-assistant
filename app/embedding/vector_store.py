"""
Vector Store Operations
ChromaDB-based vector store for document storage and semantic retrieval.
Supports both local persistent storage and optional Pinecone integration.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.embedding.embedder import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""

    def __init__(
        self,
        embedding_model: Optional[OpenAIEmbeddings] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the vector store manager.

        Args:
            embedding_model: Embedding model instance (auto-created if None).
            persist_directory: Path for ChromaDB persistence.
            collection_name: Name of the ChromaDB collection.
        """
        self.embedding_model = embedding_model or get_embedding_model()
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name
        self._vector_store: Optional[Chroma] = None

    @property
    def vector_store(self) -> Chroma:
        """Lazy-initialize and return the ChromaDB vector store."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory,
            )
            logger.info(
                f"Initialized ChromaDB: collection='{self.collection_name}', "
                f"persist_dir='{self.persist_directory}'"
            )
        return self._vector_store

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects to store.

        Returns:
            List of document IDs assigned by ChromaDB.
        """
        if not documents:
            logger.warning("No documents to add")
            return []

        logger.info(f"Adding {len(documents)} documents to vector store")
        ids = self.vector_store.add_documents(documents)
        logger.info(f"Successfully added {len(ids)} documents")
        return ids

    def create_from_documents(self, documents: list[Document]) -> Chroma:
        """
        Create a new vector store from a list of documents.

        This overwrites the existing collection with fresh data.

        Args:
            documents: Documents to populate the store with.

        Returns:
            The newly created Chroma vector store.
        """
        logger.info(
            f"Creating vector store from {len(documents)} documents"
        )
        self._vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        logger.info("Vector store created successfully")
        return self._vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[Document]:
        """
        Perform similarity search against the vector store.

        Args:
            query: Search query string.
            k: Number of results to return.
            filter_metadata: Optional metadata filter dict.

        Returns:
            List of most similar Document objects.
        """
        kwargs: dict = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata

        results = self.vector_store.similarity_search(query, **kwargs)
        logger.info(f"Similarity search for '{query[:50]}...' returned {len(results)} results")
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Perform similarity search and return results with relevance scores.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=k
        )
        return results

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a LangChain retriever from the vector store.

        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 10}).

        Returns:
            A LangChain retriever instance.
        """
        search_kwargs = search_kwargs or {"k": settings.retrieval_k}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the current collection.

        Returns:
            Dictionary with collection name and document count.
        """
        collection = self.vector_store._collection
        return {
            "collection_name": self.collection_name,
            "document_count": collection.count(),
        }

    def delete_collection(self) -> None:
        """Delete the entire collection from ChromaDB."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.vector_store.delete_collection()
        self._vector_store = None
