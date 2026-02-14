"""
Report Generator
Generates structured research reports with citations using LLM.
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.config import settings
from app.generation.prompts import (
    RESEARCH_REPORT_PROMPT,
    QUERY_REFINEMENT_PROMPT,
    FOLLOW_UP_PROMPT,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates structured research reports from retrieved documents."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the report generator.

        Args:
            model: LLM model name (defaults to settings.llm_model).
            temperature: Generation temperature (defaults to settings.llm_temperature).
        """
        self.llm = ChatOpenAI(
            model=model or settings.llm_model,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
        )
        self.report_prompt = ChatPromptTemplate.from_template(RESEARCH_REPORT_PROMPT)
        self.query_prompt = ChatPromptTemplate.from_template(QUERY_REFINEMENT_PROMPT)
        self.followup_prompt = ChatPromptTemplate.from_template(FOLLOW_UP_PROMPT)

        logger.info(f"ReportGenerator initialized with model: {model or settings.llm_model}")

    def generate(self, query: str, retrieved_docs: list[Document]) -> str:
        """
        Generate a research report from retrieved documents.

        Each document is labeled with its source for proper citation
        in the generated report.

        Args:
            query: The user's research query.
            retrieved_docs: List of relevant Document objects.

        Returns:
            Generated research report as a string.
        """
        if not retrieved_docs:
            return "⚠️ No relevant documents were retrieved. Unable to generate a report."

        # Format context with source labels for citation
        context_parts: list[str] = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = (
                doc.metadata.get("file_name")
                or doc.metadata.get("url")
                or doc.metadata.get("table_name")
                or f"Source {i}"
            )
            contextual = "Context Enriched" if doc.metadata.get("has_context") else "Original Chunk"
            header = f"[Source {i}: {source}] ({contextual})"
            context_parts.append(f"{header}\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        logger.info(f"Generating report for: '{query[:80]}...' with {len(retrieved_docs)} sources")

        chain = self.report_prompt | self.llm
        response = chain.invoke({"context": context, "query": query})

        return response.content

    def refine_query(self, query: str, num_queries: int = 3) -> list[str]:
        """
        Generate alternative search queries for better retrieval coverage.

        Args:
            query: The original user query.
            num_queries: Number of alternative queries to generate.

        Returns:
            List of refined query strings.
        """
        chain = self.query_prompt | self.llm
        response = chain.invoke({"query": query, "num_queries": num_queries})

        # Parse numbered lines
        queries: list[str] = []
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove numbering prefix (e.g., "1. ", "1) ")
                cleaned = line.lstrip("0123456789.)} ").strip()
                if cleaned:
                    queries.append(cleaned)

        return queries[:num_queries]

    def suggest_follow_ups(self, query: str, report: str) -> list[str]:
        """
        Generate follow-up research questions based on the report.

        Args:
            query: The original user query.
            report: The generated research report.

        Returns:
            List of follow-up question strings.
        """
        chain = self.followup_prompt | self.llm
        response = chain.invoke({"query": query, "report": report})

        questions: list[str] = []
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                cleaned = line.lstrip("0123456789.)} ").strip()
                if cleaned:
                    questions.append(cleaned)

        return questions[:3]
