# 🔬 RAG Research Assistant

> Multi-source research assistant with **Contextual Retrieval** (Anthropic 2024), **ChromaDB**, and **RAGAS Evaluation**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)

---

## Architecture

```
  ┌──────────────────────────────────┐
  │        Document Ingestion         │
  │   PDF / Web (Tavily) / SQL DB     │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │       Semantic Chunking           │
  │   Splits by meaning, not size     │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  ⭐ Contextual Retrieval          │
  │  LLM adds context to each chunk   │
  │  "This chunk is from section X     │
  │   of document Y, discussing Z"    │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │   Embedding + ChromaDB Store      │
  │   text-embedding-3-small (512d)   │
  └──────────────┬───────────────────┘
                 │
        ── Query Time ──
                 │
                 ▼
  ┌──────────────────────────────────┐
  │       Vector Similarity Search    │
  │   Contextual embeddings = better  │
  │   matches with fewer failures     │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │     GPT-4o-mini Report            │
  │   Citations + Follow-up Qs        │
  └──────────────────────────────────┘
```

---

## Key Techniques

### 1. Contextual Retrieval (Anthropic, 2024)
- **Problem**: Chunks lose context. "The company earned $5M in Q3" means nothing without knowing which company.
- **Solution**: Before embedding, an LLM reads the full document and generates a short context for each chunk: *"This chunk is from ACME Corp's 2024 annual report, discussing Q3 financial results."*
- **Result**: Anthropic reports **49% fewer retrieval failures**.
- **Reference**: [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval)

### 2. Semantic Chunking
- **Problem**: Fixed-size chunks split mid-sentence, breaking coherent thoughts.
- **Solution**: Uses embedding similarity between consecutive sentences to detect topic shifts.
- **Result**: Each chunk contains a complete thought unit.

### 3. RAGAS Evaluation
- **Problem**: You can't improve what you can't measure.
- **Metrics**: Faithfulness (no hallucination), Answer Relevancy, Context Precision, Context Recall.
- **Result**: Quantitative evidence that your RAG pipeline works.

---

## Quick Start

### Option 1: Standalone App (Simplest)
Use this for local demos. No separate server needed.
```bash
streamlit run app/ui/direct_app.py
```

### Option 2: Client-Server Mode
Use this for production-like setup with a separate API.
```bash
# Terminal 1: Run API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Run UI
streamlit run app/ui/streamlit_app.py
```

---

## Setup
```bash
# Clone
git clone https://github.com/shubh-vedi/rag-research-assistant.git
cd rag-research-assistant

# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env → add OPENAI_API_KEY, TAVILY_API_KEY
```

---

## Project Structure

```
rag-research-assistant/
├── app/
│   ├── config.py                       # Settings (Pydantic)
│   ├── main.py                         # FastAPI endpoints (Server Mode)
│   ├── ui/
│   │   ├── direct_app.py               # ⭐ Standalone Streamlit App
│   │   └── streamlit_app.py            # Client-Server UI
│   ├── ingestion/
│   │   ├── pdf_loader.py               # PDF → Documents
│   │   ├── web_loader.py               # Tavily → Documents
│   │   ├── db_loader.py                # SQL → Documents
│   │   └── chunker.py                  # Semantic chunking
│   ├── embedding/
│   │   ├── embedder.py                 # OpenAI embeddings
│   │   └── vector_store.py             # ChromaDB manager
│   ├── retrieval/
│   │   ├── contextual_retrieval.py     # ⭐ Contextual enrichment
│   │   └── retriever.py               # Pipeline orchestrator
│   ├── generation/
│   │   ├── prompts.py                  # Prompt templates
│   │   └── report_generator.py         # GPT-4o-mini reports
│   └── evaluation/
│       ├── ragas_eval.py               # RAGAS metrics
│       └── test_queries.json           # Test dataset
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```


## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain 1.2, FastAPI, Streamlit |
| LLM | GPT-4o-mini |
| Embeddings | text-embedding-3-small (512d) |
| Vector Store | ChromaDB |
| Key Technique | Contextual Retrieval (Anthropic 2024) |
| Web Search | Tavily |
| Evaluation | RAGAS |

---

## License

MIT License
