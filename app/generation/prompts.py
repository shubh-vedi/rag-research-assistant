"""
Prompt Templates
All prompt templates used in the RAG Research Assistant.
Centralized for easy tuning and version control.
"""

# ─── Research Report Generation ─────────────────────────────────────

RESEARCH_REPORT_PROMPT = """You are an expert research analyst. Based on the retrieved context below, \
generate a comprehensive, well-structured research report that answers the user's query.

STRICT RULES:
1. Every factual claim MUST include a citation in [Source: X] format
2. If the context does NOT contain enough information to answer fully, explicitly state what is missing
3. Do NOT hallucinate or infer facts not present in the provided context
4. Use professional, academic tone throughout

REPORT STRUCTURE:
## Executive Summary
Brief overview of findings (2-3 sentences)

## Key Findings
Bullet-point list of the most important discoveries

## Detailed Analysis
In-depth discussion organized by subtopic, with citations

## Sources Referenced
List all sources used with their identifiers

## Limitations
Note any gaps in the available information

---

RETRIEVED CONTEXT:
{context}

---

USER QUERY: {query}

Generate the research report now:"""


# ─── Query Refinement ───────────────────────────────────────────────

QUERY_REFINEMENT_PROMPT = """You are a search query optimizer. Given a user's research question, \
generate {num_queries} alternative search queries that would help find comprehensive information.

Each query should:
1. Cover a different angle or aspect of the topic
2. Use different keywords or phrasings
3. Range from broad to specific

ORIGINAL QUERY: {query}

Generate {num_queries} refined search queries (one per line, numbered):"""


# ─── Context Summarization ──────────────────────────────────────────

CONTEXT_SUMMARY_PROMPT = """Summarize the following text chunk while preserving all key facts, \
figures, and relationships. The summary will be used for research report generation.

Keep the summary concise but information-dense. Do not omit any factual claims.

TEXT:
{text}

CONCISE SUMMARY:"""


# ─── Follow-Up Question Generation ──────────────────────────────────

FOLLOW_UP_PROMPT = """Based on the research report and original query below, suggest 3 follow-up \
questions that would deepen the research.

ORIGINAL QUERY: {query}

REPORT:
{report}

Suggest 3 follow-up questions (one per line, numbered):"""
