"""
PDF Document Loader
Loads PDF files and extracts text content with enriched metadata.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF file and return a list of Document objects (one per page).

    Each document is enriched with metadata:
      - source_type: "pdf"
      - file_name: basename of the file
      - page_number: 1-indexed page number
      - total_pages: total number of pages in the PDF

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of LangChain Document objects with page content and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    logger.info(f"Loading PDF: {path.name}")
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    # Enrich metadata
    total_pages = len(pages)
    for i, page in enumerate(pages):
        page.metadata.update({
            "source_type": "pdf",
            "file_name": path.name,
            "page_number": i + 1,
            "total_pages": total_pages,
        })

    logger.info(f"Loaded {total_pages} pages from {path.name}")
    return pages


def load_multiple_pdfs(file_paths: list[str]) -> list[Document]:
    """
    Load multiple PDF files and return a combined list of documents.

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        Combined list of Document objects from all PDFs.
    """
    all_documents: list[Document] = []
    for file_path in file_paths:
        try:
            docs = load_pdf(file_path)
            all_documents.extend(docs)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {file_path}: {e}")
    return all_documents
