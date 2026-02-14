"""
Streamlit UI for the RAG Research Assistant.
Clean interface for document ingestion, research queries, and evaluation.
"""

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stMetric > div { background: #1e1e2e; border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


def api_request(endpoint: str, method: str = "GET", **kwargs):
    """Helper for API requests with error handling."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=120)
        elif method == "POST":
            resp = requests.post(url, timeout=120, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API server. Is it running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e.response.text}")
        return None


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header">🔬 RAG Assistant</p>', unsafe_allow_html=True)
    st.caption("Contextual Retrieval Only")

    st.divider()

    health = api_request("/health")
    if health:
        st.success("✅ API Connected")
        if health.get("vector_store"):
            vs = health["vector_store"]
            st.metric("Documents Indexed", vs.get("document_count", 0))

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
                result = api_request(
                    "/research",
                    method="POST",
                    json={
                        "query": query,
                        "top_k": top_k,
                        "include_web": include_web,
                    },
                )

            if result:
                # Report
                st.markdown("### 📄 Report")
                st.markdown(result["report"])

                # Sources
                with st.expander(f"📎 Sources ({len(result.get('sources', []))})", expanded=False):
                    for i, src in enumerate(result.get("sources", []), 1):
                        contextual = "🧠" if src.get("contextual") else ""
                        st.markdown(
                            f"**{i}.** {contextual} [{src['source_type']}] {src['source']}"
                        )

                # Follow-ups
                follow_ups = result.get("follow_up_questions", [])
                if follow_ups:
                    st.markdown("### 💡 Follow-up Questions")
                    for fq in follow_ups:
                        st.markdown(f"- {fq}")


# ── Tab 2: Ingestion ────────────────────────────────────────────────
with tab_ingest:
    st.header("Document Ingestion")
    st.info("📌 Documents are automatically enriched with **Contextual Retrieval** during ingestion.")

    col_pdf, col_web = st.columns(2)

    with col_pdf:
        st.subheader("📄 PDF Upload")
        uploaded = st.file_uploader("Upload a PDF document", type=["pdf"])
        if uploaded and st.button("Process PDF", type="primary"):
            with st.spinner("Ingesting + enriching with context..."):
                response = api_request(
                    "/ingest", method="POST",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                )
            if response:
                st.success(
                    f"✅ {response['message']}\n\n"
                    f"Chunks: {response['chunks_created']} | "
                    f"Contextually enriched: {'✅' if response.get('enriched') else '❌'}"
                )

    with col_web:
        st.subheader("🌐 Web Search")
        web_query = st.text_input("Search query:")
        web_count = st.slider("Max results", 1, 10, 3)
        if web_query and st.button("Ingest from Web", type="primary"):
            with st.spinner("Searching and enriching..."):
                response = api_request(
                    "/ingest/web", method="POST",
                    json={"query": web_query, "max_results": web_count},
                )
            if response:
                st.success(f"✅ {response['message']}")


# ── Tab 3: Evaluation ───────────────────────────────────────────────
with tab_eval:
    st.header("RAGAS Evaluation")
    st.caption("Evaluate RAG quality using industry-standard metrics")

    st.markdown("""
    | Metric | What it Measures |
    |--------|-----------------|
    | **Faithfulness** | Is the answer grounded in the retrieved context? |
    | **Answer Relevancy** | Does the answer address the question? |
    | **Context Precision** | Are the retrieved chunks relevant? |
    | **Context Recall** | Did we retrieve all needed information? |
    """)

    if st.button("▶️ Run Evaluation", type="primary"):
        with st.spinner("Running RAGAS evaluation..."):
            result = api_request("/evaluate", method="POST")

        if result:
            cols = st.columns(4)
            metrics = [
                ("Faithfulness", result["scores"].get("faithfulness", 0)),
                ("Answer Relevancy", result["scores"].get("answer_relevancy", 0)),
                ("Context Precision", result["scores"].get("context_precision", 0)),
                ("Context Recall", result["scores"].get("context_recall", 0)),
            ]
            for col, (name, score) in zip(cols, metrics):
                col.metric(name, f"{score:.3f}")
