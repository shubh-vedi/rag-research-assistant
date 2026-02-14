"""
Web Search Loader
Uses Tavily API to search the web and return results as LangChain Documents.
"""

import logging
import os

from langchain_core.documents import Document
from tavily import TavilyClient

from app.config import settings

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 5) -> list[Document]:
    """
    Search the web using Tavily and return results as Document objects.

    Each document includes metadata:
      - source_type: "web"
      - url: source URL
      - title: page title
      - score: relevance score from Tavily

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of LangChain Document objects with web content.

    Raises:
        ValueError: If no API key is configured.
    """
    api_key = settings.tavily_api_key or os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Tavily API key not configured. Set TAVILY_API_KEY in your .env file."
        )

    logger.info(f"Searching web for: '{query}' (max_results={max_results})")
    client = TavilyClient(api_key=api_key)

    results = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=False,
    )

    documents: list[Document] = []
    for result in results.get("results", []):
        doc = Document(
            page_content=result.get("content", ""),
            metadata={
                "source_type": "web",
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "score": result.get("score", 0.0),
            },
        )
        documents.append(doc)

    logger.info(f"Retrieved {len(documents)} web results for: '{query}'")
    return documents


def search_web_batch(queries: list[str], max_results_per_query: int = 3) -> list[Document]:
    """
    Search the web for multiple queries and return combined results.

    Args:
        queries: List of search query strings.
        max_results_per_query: Maximum results per query.

    Returns:
        Combined list of Document objects from all queries.
    """
    all_documents: list[Document] = []
    for query in queries:
        try:
            docs = search_web(query, max_results=max_results_per_query)
            all_documents.extend(docs)
        except Exception as e:
            logger.warning(f"Web search failed for '{query}': {e}")
    return all_documents
