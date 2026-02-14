"""
Standalone Streamlit App
Runs the RAG Research Assistant directly without a separate FastAPI backend.
Good for local demos and debugging.
"""

import logging
import shutil
from pathlib import Path
import streamlit as st

# Direct backend imports
from app.config import settings
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.web_loader import search_web
from app.ingestion.chunker import semantic_chunk
from app.embedding.embedder import get_embedding_model
from app.embedding.vector_store import VectorStoreManager
from app.retrieval.contextual_retrieval import ContextualRetrieval
from app.retrieval.retriever import ResearchRetriever
from app.generation.report_generator import ReportGenerator

# ── Logging Setup ────────────────────────────────────────────────────
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RAG Research Assistant (Standalone)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #FF4B2B 0%, #FF416C 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stMetric > div { background: #1e1e2e; border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Backend Initialization ───────────────────────────────────────────
@st.cache_resource
def get_backend():
    """Initialize backend components once."""
    settings.ensure_directories()
    
    embedding_model = get_embedding_model()
    vector_store_manager = VectorStoreManager(embedding_model=embedding_model)
    contextual_retrieval = ContextualRetrieval()
    generator = ReportGenerator()
    
    return vector_store_manager, contextual_retrieval, generator

vector_store_manager, contextual_retrieval, generator = get_backend()


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header">⚡ Standalone Mode</p>', unsafe_allow_html=True)
    st.caption("Direct Backend Connection")

    st.divider()

    # Show stats directly from vector store
    try:
        stats = vector_store_manager.get_collection_stats()
        st.success("✅ Backend Active")
        st.metric("Documents", stats.get("document_count", 0))
    except Exception as e:
        st.error(f"❌ Backend Error: {e}")

    st.divider()
    st.markdown("### Pipeline")
    st.markdown("""
    **Ingest:**
    1. 📄 Load (PDF / Web)
    2. ✂️ Semantic Chunking
    3. 🧠 **Contextual Enrichment**
    4. 💾 Embed + Store (ChromaDB)

    **Query:**
    1. 🔍 Vector Search (Contextual)
    2. 📝 GPT-4o-mini Report
    """)


# ── Main Tabs ────────────────────────────────────────────────────────
tab_research, tab_ingest, tab_eval = st.tabs([
    "🔬 Research", "📤 Ingest Documents", "📊 Evaluation"
])


# ── Tab 1: Research ──────────────────────────────────────────────────
with tab_research:
    st.header("Research Query")

    query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., What are the key contributions of the Transformer architecture?",
        height=100,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        top_k = st.slider("Number of results", 3, 15, 5)
    with col2:
        include_web = st.checkbox("Include Web Search", value=False)

    if st.button("🔍 Research", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a research question.")
        else:
            with st.spinner("Running RAG pipeline..."):
                try:
                    # 1. Optional Web Search
                    if include_web:
                        with st.status("🌐 Searching web..."):
                            web_docs = search_web(query, max_results=3)
                            web_chunks = semantic_chunk(web_docs)
                            full_text = "\n\n".join(doc.page_content for doc in web_docs)
                            enriched = contextual_retrieval.enrich_chunks(web_chunks, full_text)
                            vector_store_manager.add_documents(enriched)
                    
                    # 2. Retrieve
                    retriever = ResearchRetriever(vector_store_manager)
                    docs = retriever.retrieve(query, top_k=top_k)

                    if not docs:
                        st.error("No relevant documents found. Ingrest some documents first!")
                    else:
                        # 3. Generate
                        report = generator.generate(query, docs)
                        follow_ups = generator.suggest_follow_ups(query, report)

                        # Display Report
                        st.markdown("### 📄 Report")
                        st.markdown(report)

                        # Display Sources
                        with st.expander(f"📎 Sources ({len(docs)})", expanded=False):
                            for i, doc in enumerate(docs, 1):
                                source = (
                                    doc.metadata.get("file_name")
                                    or doc.metadata.get("url")
                                    or doc.metadata.get("table_name")
                                    or "unknown"
                                )
                                contextual = "🧠" if doc.metadata.get("has_context") else ""
                                st.markdown(f"**{i}.** {contextual} [{source}]")
                                st.caption(doc.page_content[:200] + "...")

                        # Follow-ups
                        if follow_ups:
                            st.markdown("### 💡 Follow-up Questions")
                            for fq in follow_ups:
                                st.markdown(f"- {fq}")

                except Exception as e:
                    st.error(f"Error during research: {e}")


# ── Tab 2: Ingestion ────────────────────────────────────────────────
with tab_ingest:
    st.header("Document Ingestion")
    st.info("📌 Documents are automatically enriched with **Contextual Retrieval**.")

    col_pdf, col_web = st.columns(2)

    with col_pdf:
        st.subheader("📄 PDF Upload")
        uploaded = st.file_uploader("Upload a PDF document", type=["pdf"])
        if uploaded and st.button("Process PDF", type="primary"):
            with st.spinner("Ingesting + enriching with context..."):
                try:
                    # Save file
                    upload_path = Path(settings.upload_dir) / uploaded.name
                    with open(upload_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    
                    # Process
                    documents = load_pdf(str(upload_path))
                    chunks = semantic_chunk(documents)
                    full_text = "\n\n".join(doc.page_content for doc in documents)
                    enriched_chunks = contextual_retrieval.enrich_chunks(chunks, full_text)
                    vector_store_manager.add_documents(enriched_chunks)

                    st.success(
                        f"✅ Successfully ingested '{uploaded.name}'\n\n"
                        f"Chunks: {len(chunks)} | Enriched: {len(enriched_chunks)}"
                    )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    with col_web:
        st.subheader("🌐 Web Search")
        web_query = st.text_input("Search query:")
        web_count = st.slider("Max results", 1, 10, 3)
        if web_query and st.button("Ingest from Web", type="primary"):
            with st.spinner("Searching and enriching..."):
                try:
                    documents = search_web(web_query, max_results=web_count)
                    chunks = semantic_chunk(documents)
                    full_text = "\n\n".join(doc.page_content for doc in documents)
                    enriched_chunks = contextual_retrieval.enrich_chunks(chunks, full_text)
                    vector_store_manager.add_documents(enriched_chunks)

                    st.success(f"✅ Ingested {len(documents)} results for '{web_query}'")
                except Exception as e:
                    st.error(f"Web ingestion failed: {e}")


# ── Tab 3: Evaluation ───────────────────────────────────────────────
with tab_eval:
    st.header("RAGAS Evaluation")
    
    if st.button("▶️ Run Evaluation", type="primary"):
        with st.spinner("Running RAGAS evaluation..."):
            try:
                from app.evaluation.ragas_eval import run_evaluation_from_file
                retriever = ResearchRetriever(vector_store_manager)
                scores = run_evaluation_from_file(retriever=retriever, generator=generator)

                if scores:
                    cols = st.columns(4)
                    metrics = [
                        ("Faithfulness", scores.get("faithfulness", 0)),
                        ("Answer Relevancy", scores.get("answer_relevancy", 0)),
                        ("Context Precision", scores.get("context_precision", 0)),
                        ("Context Recall", scores.get("context_recall", 0)),
                    ]
                    for col, (name, score) in zip(cols, metrics):
                        col.metric(name, f"{score:.3f}")
                else:
                    st.error("Evaluation failed (no scores returned)")
            except Exception as e:
                st.error(f"Evaluation error: {e}")
