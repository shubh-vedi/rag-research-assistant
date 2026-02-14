"""
FastAPI Entry Point — RAG Research Assistant
Provides REST API for document ingestion, research queries, and evaluation.
"""

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.web_loader import search_web
from app.ingestion.chunker import semantic_chunk
from app.embedding.embedder import get_embedding_model
from app.embedding.vector_store import VectorStoreManager
from app.retrieval.contextual_retrieval import ContextualRetrieval
from app.retrieval.retriever import ResearchRetriever
from app.generation.report_generator import ReportGenerator

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global State ─────────────────────────────────────────────────────
vector_store_manager: VectorStoreManager | None = None
contextual_retrieval: ContextualRetrieval | None = None
generator: ReportGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup, clean up on shutdown."""
    global vector_store_manager, contextual_retrieval, generator

    settings.ensure_directories()

    vector_store_manager = VectorStoreManager(
        embedding_model=get_embedding_model(),
    )
    contextual_retrieval = ContextualRetrieval()
    generator = ReportGenerator()
    logger.info("🚀 RAG Research Assistant started")

    yield

    logger.info("👋 RAG Research Assistant shutting down")


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Research Assistant",
    description=(
        "Multi-source research assistant with Contextual Retrieval (Anthropic 2024), "
        "and RAGAS evaluation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────
class ResearchRequest(BaseModel):
    query: str
    top_k: int = 5
    include_web: bool = False
    web_results: int = 3


class ResearchResponse(BaseModel):
    report: str
    sources: list[dict]
    follow_up_questions: list[str]


class IngestResponse(BaseModel):
    message: str
    chunks_created: int
    enriched: bool


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class EvalResponse(BaseModel):
    scores: dict
    num_queries: int


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = None
    if vector_store_manager:
        try:
            stats = vector_store_manager.get_collection_stats()
        except Exception:
            stats = {"status": "collection not initialized"}

    return {"status": "healthy", "vector_store": stats}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.

    Pipeline: Upload → Parse → Chunk → Contextual Enrichment → Embed → Store
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file
    upload_path = Path(settings.upload_dir) / file.filename
    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Load and chunk
    try:
        documents = load_pdf(str(upload_path))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}")

    chunks = semantic_chunk(documents)

    # Contextual Retrieval: enrich chunks with document context
    full_text = "\n\n".join(doc.page_content for doc in documents)
    enriched_chunks = contextual_retrieval.enrich_chunks(chunks, full_text)

    # Embed and store
    vector_store_manager.add_documents(enriched_chunks)

    logger.info(
        f"Ingested '{file.filename}': {len(documents)} pages → "
        f"{len(chunks)} chunks → {len(enriched_chunks)} enriched"
    )

    return IngestResponse(
        message=f"Successfully ingested '{file.filename}'",
        chunks_created=len(enriched_chunks),
        enriched=True,
    )


@app.post("/ingest/web")
async def ingest_web(request: WebSearchRequest):
    """Search the web via Tavily and ingest results."""
    documents = search_web(request.query, max_results=request.max_results)
    chunks = semantic_chunk(documents)

    # Contextual enrichment
    full_text = "\n\n".join(doc.page_content for doc in documents)
    enriched_chunks = contextual_retrieval.enrich_chunks(chunks, full_text)

    vector_store_manager.add_documents(enriched_chunks)

    return IngestResponse(
        message=f"Ingested {len(documents)} web results for '{request.query}'",
        chunks_created=len(enriched_chunks),
        enriched=True,
    )


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Run a research query through the RAG pipeline.

    Pipeline: Vector Search → LLM Report Generation
    """
    # Optionally augment with live web results
    if request.include_web:
        web_docs = search_web(request.query, max_results=request.web_results)
        web_chunks = semantic_chunk(web_docs)
        full_text = "\n\n".join(doc.page_content for doc in web_docs)
        enriched = contextual_retrieval.enrich_chunks(web_chunks, full_text)
        vector_store_manager.add_documents(enriched)

    # Retrieve
    retriever = ResearchRetriever(vector_store_manager)
    docs = retriever.retrieve(request.query, top_k=request.top_k)

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No relevant documents found. Try ingesting documents first via /ingest.",
        )

    # Generate report
    report = generator.generate(request.query, docs)
    follow_ups = generator.suggest_follow_ups(request.query, report)

    # Collect source metadata
    sources = []
    for doc in docs:
        sources.append({
            "source_type": doc.metadata.get("source_type", "unknown"),
            "source": (
                doc.metadata.get("file_name")
                or doc.metadata.get("url")
                or doc.metadata.get("table_name")
                or "unknown"
            ),
            "contextual": doc.metadata.get("has_context", False),
            "preview": doc.page_content[:200] + "...",
        })

    return ResearchResponse(
        report=report,
        sources=sources,
        follow_up_questions=follow_ups,
    )


@app.post("/evaluate", response_model=EvalResponse)
async def run_evaluation():
    """Run RAGAS evaluation on the test dataset."""
    from app.evaluation.ragas_eval import run_evaluation_from_file

    retriever = ResearchRetriever(vector_store_manager)
    scores = run_evaluation_from_file(retriever=retriever, generator=generator)

    if scores is None:
        raise HTTPException(status_code=500, detail="Evaluation failed")

    return EvalResponse(scores=scores, num_queries=5)


@app.get("/documents/stats")
async def document_stats():
    """Get statistics about ingested documents."""
    store_stats = {}
    if vector_store_manager:
        try:
            store_stats = vector_store_manager.get_collection_stats()
        except Exception:
            store_stats = {"status": "unavailable"}

    return {
        "vector_store": store_stats,
        "technique": "Contextual Retrieval (Anthropic 2024)",
    }
