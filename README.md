# Self-RAG University Course Advisory Agent

A Self-Reflective RAG (Retrieval-Augmented Generation) agent for XYZ National University, built with LangGraph, Mistral AI, and FAISS. The agent adaptively decides when to retrieve, grades every retrieved chunk for relevance, falls back to web search when the knowledge base has no relevant results, and self-checks its own responses for hallucinations before delivery.

**AI407L Capstone Lab — Spring 2026 Final Exam Part B**

---

## Architecture

```
User Query
    |
    v
[decide_retrieval]  -- LLM decides: retrieve or answer directly?
    |
    +-- DIRECT  --> [direct_answer] --> Final Answer
    |
    +-- RETRIEVE --> [retrieve_docs]  (FAISS similarity search)
                          |
                    [grade_docs]  -- each chunk graded relevant/irrelevant
                          |
              +-[relevant]--> [generate_response]
              |                       |
              +-[all irrelevant]--> [web_search] --> [generate_response]
                                                          |
                                               [hallucination_check]
                                                          |
                                      +-[grounded]------> Final Answer
                                      +-[retry < 2]-----> [generate_response]
                                      +-[max retries]---> [disclaimer]
```

**8 LangGraph nodes** — `decide_retrieval`, `retrieve_docs`, `grade_docs`, `web_search`, `generate_response`, `direct_answer`, `hallucination_check`, `disclaimer`

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM (decisions + generation) | Mistral `mistral-small-latest` |
| Embeddings | Mistral `mistral-embed` (1024-dim) |
| Vector Store | FAISS (local, no server needed) |
| Orchestration | LangGraph 1.0.10 StateGraph |
| Web Search Fallback | DuckDuckGo (no API key needed) |
| Web Interface | Flask + vanilla HTML/CSS/JS |
| PDF Loading | PyPDFLoader (langchain-community) |
| Chunking | RecursiveCharacterTextSplitter (1000/200) |

---

## Knowledge Base

Five university PDF documents, chunked and embedded into a local FAISS index (`university_kb/`):

- `CS_Department_Catalog.pdf` — CS courses, prerequisites, credit hours
- `EE_Department_Catalog.pdf` — Electrical Engineering courses
- `BBA_Department_Catalog.pdf` — Business Administration courses
- `University_Academic_Policies.pdf` — GPA rules, grading, withdrawal policy, fees
- `Faculty_Directory.pdf` — Faculty names, offices, emails, specializations

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/haris-ismail/self-rag-university-advisor.git
cd self-rag-university-advisor
pip install langchain langgraph langchain-community langchain-mistralai langchain-text-splitters faiss-cpu duckduckgo-search pypdf flask python-dotenv
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Mistral API key
# Get one free at: https://console.mistral.ai
```

### 3. Run the web interface

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 4. Or run from the terminal

```bash
# Interactive mode
python self_rag_agent.py

# Single query
python self_rag_agent.py --query "What are the prerequisites for CS302?"

# Rebuild the knowledge base from PDFs
python ingest.py --rebuild
```

---

## Test Suite

Run all 5 Self-RAG evaluation scenarios:

```bash
python run_tests.py
```

| # | Scenario | Expected Pipeline Path |
|---|----------|----------------------|
| 1 | Greeting / general knowledge | `direct_answer` — no retrieval |
| 2 | ML course prerequisites | `retrieve -> grade -> generate -> hallucination_check` |
| 3 | Out-of-domain query (stock price) | `retrieve -> all irrelevant -> web_search -> generate -> check` |
| 4 | Exhaustive multi-department request | `retrieve -> web_search -> hallucination retry -> grounded` |
| 5 | Cross-department faculty office query | `retrieve -> web_search -> generate -> check` |

Results are written to `evaluation_results.md` with full decision traces.

---

## Project Structure

```
self-rag-university-advisor/
  app.py                  -- Flask web server (http://localhost:5000)
  ingest.py               -- Build FAISS knowledge base from PDFs
  tools.py                -- 4 @tool functions with Pydantic schemas
  graph.py                -- LangGraph Self-RAG StateGraph (8 nodes)
  self_rag_agent.py       -- Interactive CLI entry point
  run_tests.py            -- Automated 5-scenario test runner
  evaluation_results.md   -- Test execution traces (auto-generated)
  HOW_TO_RUN.md           -- Detailed run instructions
  university_kb/          -- Pre-built FAISS index (23 chunks)
  templates/
    index.html            -- Chat UI frontend
  .env.example            -- API key template
```

---

## Self-RAG Key Features

- **Adaptive retrieval** — LLM decides per-query whether retrieval is needed at all
- **Per-document grading** — every retrieved chunk is individually scored relevant/irrelevant
- **Web search fallback** — DuckDuckGo search triggers automatically when all chunks are irrelevant
- **Hallucination self-check** — generated response is verified against source context before delivery
- **Retry loop** — up to 2 retries with stricter prompting; disclaimer issued on max retries
- **Full decision trace** — every pipeline decision is logged and surfaced in the UI

---

## License

MIT
