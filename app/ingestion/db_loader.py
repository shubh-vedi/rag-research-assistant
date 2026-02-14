"""
Database Document Loader
Connects to SQL databases and loads table rows as LangChain Documents.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from sqlalchemy import create_engine, text, inspect

from app.config import settings

logger = logging.getLogger(__name__)


def load_from_database(
    table_name: str,
    content_columns: list[str],
    metadata_columns: Optional[list[str]] = None,
    where_clause: Optional[str] = None,
    db_url: Optional[str] = None,
    limit: int = 1000,
) -> list[Document]:
    """
    Load rows from a SQL database table as Document objects.

    The content of specified columns is concatenated to form the page_content.
    Additional columns can be included as metadata.

    Args:
        table_name: Name of the database table.
        content_columns: Column names whose values form the document content.
        metadata_columns: Optional column names to include as metadata.
        where_clause: Optional SQL WHERE clause (without 'WHERE' keyword).
        db_url: Database connection URL (defaults to settings.database_url).
        limit: Maximum number of rows to load.

    Returns:
        List of LangChain Document objects.

    Raises:
        ValueError: If the table or columns do not exist.
    """
    db_url = db_url or settings.database_url
    engine = create_engine(db_url)

    # Validate table exists
    inspector = inspect(engine)
    available_tables = inspector.get_table_names()
    if table_name not in available_tables:
        raise ValueError(
            f"Table '{table_name}' not found. Available tables: {available_tables}"
        )

    # Validate columns exist
    table_columns = [col["name"] for col in inspector.get_columns(table_name)]
    all_requested = content_columns + (metadata_columns or [])
    missing = [col for col in all_requested if col not in table_columns]
    if missing:
        raise ValueError(
            f"Columns not found in '{table_name}': {missing}. "
            f"Available: {table_columns}"
        )

    # Build query
    select_cols = content_columns + (metadata_columns or [])
    columns_sql = ", ".join(select_cols)
    query = f"SELECT {columns_sql} FROM {table_name}"  # noqa: S608

    if where_clause:
        query += f" WHERE {where_clause}"
    query += f" LIMIT {limit}"

    logger.info(f"Loading documents from table: {table_name}")

    documents: list[Document] = []
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.mappings().all()

        for i, row in enumerate(rows):
            # Concatenate content columns
            content_parts = []
            for col in content_columns:
                value = row.get(col, "")
                if value:
                    content_parts.append(f"{col}: {value}")
            page_content = "\n".join(content_parts)

            # Build metadata
            metadata = {
                "source_type": "database",
                "table_name": table_name,
                "row_index": i,
            }
            if metadata_columns:
                for col in metadata_columns:
                    metadata[col] = row.get(col)

            documents.append(
                Document(page_content=page_content, metadata=metadata)
            )

    logger.info(f"Loaded {len(documents)} documents from {table_name}")
    return documents


def list_available_tables(db_url: Optional[str] = None) -> list[str]:
    """
    List all available tables in the configured database.

    Args:
        db_url: Database connection URL (defaults to settings.database_url).

    Returns:
        List of table names.
    """
    db_url = db_url or settings.database_url
    engine = create_engine(db_url)
    inspector = inspect(engine)
    return inspector.get_table_names()
