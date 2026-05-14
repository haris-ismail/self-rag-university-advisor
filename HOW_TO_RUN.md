# HOW TO RUN — Self-RAG University Course Advisory Agent
# AI407L Final Exam — Part B
# XYZ National University

---

## Project Structure

```
PART B SEPARATE/
  ingest.py            — Step 1: build the knowledge base from 5 PDFs
  tools.py             — Tool definitions (@tool + Pydantic schemas)
  graph.py             — LangGraph Self-RAG StateGraph (8 nodes)
  self_rag_agent.py    — Step 3: run the interactive agent
  run_tests.py         — Step 2: run all 5 test cases + generate report
  evaluation_results.md — Auto-generated test report (real traces)
  university_kb/       — FAISS vector index (already built, 23 chunks)
  .env                 — API keys
  requirements_partb.txt
```

---

## Prerequisites

### 1. Python version
Requires **Python 3.12**.

### 2. Install dependencies
All main packages are already installed in the project environment. Only one extra package is needed:

```bash
pip install mistralai duckduckgo-search faiss-cpu
```

### 3. API key
Copy `.env.example` to `.env` and add your Mistral API key:
```
cp .env.example .env
# then edit .env:
MISTRAL_API_KEY=your_mistral_api_key_here
```
Get a free key at: https://console.mistral.ai

---

## Step-by-Step Run Plan

### STEP 1 — Build the Knowledge Base (run once)

The `university_kb/` folder is already included with a pre-built FAISS index.
Skip this step unless you want to rebuild from scratch.

```bash
cd "PART B SEPARATE"
python ingest.py
```

To force a full rebuild:
```bash
python ingest.py --rebuild
```

**What it does:**
- Loads all 5 PDFs from `Final/Data_share/`
- Splits them into 23 chunks (chunk_size=1000, overlap=200)
- Embeds each chunk using Mistral `mistral-embed` model
- Saves the FAISS index to `university_kb/`

**Expected output:**
```
[INGEST] Building knowledge base from scratch...
  Loading: CS_Department_Catalog.pdf    -> 3 pages -> 7 chunks
  Loading: EE_Department_Catalog.pdf    -> 2 pages -> 4 chunks
  Loading: BBA_Department_Catalog.pdf   -> 2 pages -> 4 chunks
  Loading: University_Academic_Policies.pdf -> 3 pages -> 6 chunks
  Loading: Faculty_Directory.pdf        -> 1 pages -> 2 chunks
[INGEST] Total chunks: 23
[INGEST] Done. 23 chunks indexed and saved to university_kb/
```

---

### STEP 2 — Run All 5 Test Cases

```bash
cd "PART B SEPARATE"
python run_tests.py
```

**What it does:**
- Compiles the LangGraph Self-RAG pipeline
- Runs 5 predefined test scenarios (required by the exam rubric)
- Prints the full decision trace for each test
- Overwrites `evaluation_results.md` with actual execution traces

**Test scenarios covered:**

| # | Scenario | Expected Pipeline Path |
|---|----------|----------------------|
| 1 | Greeting / GPA general knowledge | `direct_answer` — no retrieval at all |
| 2 | ML course prerequisites (CS-specific) | `retrieve -> grade -> generate -> hallucination_check` |
| 3 | Stock price (out-of-domain query) | `retrieve -> all irrelevant -> web_search -> generate -> check` |
| 4 | Exhaustive multi-department request | `retrieve -> web_search -> generate -> check` |
| 5 | Cross-department faculty office query | `retrieve -> grade -> generate -> check` |

**Expected output:**
```
TEST SUMMARY: 5/5 test cases completed successfully.
[REPORT] evaluation_results.md written to ...
```

---

### STEP 3 — Run the Interactive Agent

```bash
cd "PART B SEPARATE"
python self_rag_agent.py
```

The agent starts in interactive mode. Type any question and press Enter.

**Example session:**
```
You: What courses does the CS department offer?

[ADAPTIVE RETRIEVAL] -> RETRIEVE from KB
[RETRIEVAL] Retrieved 4 chunks from knowledge base.
[GRADING] 2/4 relevant.
[GENERATE] Response produced.
[HALLUCINATION CHECK] GROUNDED.

Agent: The CS Department offers the following courses:
  CS-101: Introduction to Programming (3 credits, no prerequisites)
  CS-102: Data Structures & Algorithms (3 credits, prereq: CS-101)
  ...

You: trace          <- type 'trace' to see the full decision trail
You: quit           <- exit
```

**Other run modes:**
```bash
# Single query, non-interactive
python self_rag_agent.py --query "What are the fees for the EE department?"

# Rebuild KB then start interactive mode
python self_rag_agent.py --ingest
```

---

## Self-RAG Pipeline Decision Flow

```
User Query
    |
    v
[decide_retrieval]  <-- LLM decides: retrieve or answer directly?
    |
    +-- [DIRECT] --> [direct_answer] --> Final Answer
    |
    +-- [RETRIEVE] --> [retrieve_docs]  <-- searches FAISS (Mistral embeddings)
                            |
                            v
                      [grade_docs]  <-- each chunk graded relevant/irrelevant
                            |
                +-- [relevant docs] --> [generate_response]
                |                              |
                +-- [all irrelevant] --> [web_search] --> [generate_response]
                                                               |
                                                               v
                                                    [hallucination_check]
                                                               |
                                            +-- [grounded] --> Final Answer
                                            |
                                            +-- [hallucinated, retries left]
                                            |           --> [generate_response] (retry)
                                            |
                                            +-- [hallucinated, max retries=2]
                                                        --> [disclaimer] --> Final Answer
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: mistralai` | `pip install mistralai` |
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| `ModuleNotFoundError: duckduckgo_search` | `pip install duckduckgo-search` |
| Mistral 504 timeout | Wait 30s and retry — transient server issue |
| `university_kb/ not found` | Run `python ingest.py --rebuild` first |
| PDFs not found during ingest | Check that `AI475/Final/Data_share/` contains all 5 PDFs |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (all decisions + generation) | Mistral `mistral-small-latest` |
| Embeddings | Mistral `mistral-embed` (1024-dim) |
| Vector Store | FAISS (local, no server needed) |
| Orchestration | LangGraph 1.0.10 StateGraph |
| Web Search Fallback | DuckDuckGo (no API key needed) |
| PDF Loading | PyPDFLoader (langchain-community) |
| Chunking | RecursiveCharacterTextSplitter (1000/200) |

---

*AI407L Capstone Lab — Spring 2026 Final Exam Part B*
