# Evaluation Results — Self-RAG University Course Advisory Agent

**Course:** AI407L Capstone Lab — Spring 2026  
**Part:** B — Self-RAG Agent  
**Generated:** 2026-05-14 19:07:28  
**Model:** mistral-small-latest
**Vector Store:** FAISS (university_kb/)  

---

## Overview

This document records the execution of all 5 required Self-RAG test scenarios. Each section shows the query, the expected pipeline path, the actual decision trace captured during execution, and the agent's final response.

### Self-RAG Decision Paths

```
START -> decide_retrieval
          +-[direct]---> direct_answer -> END
          +-[retrieve]-> retrieve_docs -> grade_docs
                                          +-[relevant docs]---> generate_response
                                          +-[no relevant]-----> web_search -> generate_response
                                                                               ↓
                                                                   hallucination_check
                                                                       +-[grounded]-----> END
                                                                       +-[retry]--------> generate_response
                                                                       +-[max retries]--> disclaimer -> END
```

---

## Test Case 1: A query where retrieval is NOT needed

### Query

> Hello! Can you tell me what GPA stands for?

### Expected Behavior

**Expected pipeline path:** `direct_answer`

The query is a greeting combined with a general-knowledge question. 'GPA' (Grade Point
Average) is common knowledge that does not require searching the university knowledge
base. The agent should classify this as 'direct' and answer immediately without any
retrieval.

### Actual Behavior

**Status:** ✅ Completed successfully
  - Retrieval needed: `False`
  - Web search used: `False`
  - Chunks retrieved: `0`
  - Chunks graded relevant: `0`
  - Hallucination detected: `False`
  - Retry count: `0`

**Actual pipeline path:**
> `decide_retrieval[direct] -> direct_answer -> END`

**Decision Trace:**

```
  1. [ADAPTIVE RETRIEVAL] 'Hello! Can you tell me what GPA stands for?' -> DIRECT answer (no retrieval)
  2. [DIRECT ANSWER] Responded from general knowledge. No KB retrieval performed.
```

**Final Response:**

> Hello! Great question! GPA stands for **Grade Point Average**. It's a standard way of measuring academic performance by converting letter grades into a numerical average (usually on a 4.0 scale). Most universities use it to assess a student's overall achievement in their courses.

---

## Test Case 2: A query where retrieval IS needed and documents are relevant

### Query

> What are the prerequisites for the Machine Learning course in the CS department?

### Expected Behavior

**Expected pipeline path:** `retrieve_docs -> grade_docs -> generate_response -> hallucination_check -> END`

This is a specific university question about a CS course's prerequisites. The agent should
classify this as 'retrieve', fetch chunks from CS_Department_Catalog.pdf, grade them as
relevant, generate an answer grounded in those chunks, pass the hallucination check, and
deliver the answer.

### Actual Behavior

**Status:** ✅ Completed successfully
  - Retrieval needed: `True`
  - Web search used: `False`
  - Chunks retrieved: `4`
  - Chunks graded relevant: `1`
  - Hallucination detected: `False`
  - Retry count: `0`

**Actual pipeline path:**
> `decide_retrieval[retrieve] -> retrieve_docs(4 docs) -> grade_docs(1 relevant) -> generate_response -> hallucination_check[grounded/disclaimer] -> END`

**Decision Trace:**

```
  1. [ADAPTIVE RETRIEVAL] 'What are the prerequisites for the Machine Learning course in the CS department?' -> RETRIEVE from KB
  2. [RETRIEVAL] Retrieved 4 chunks from knowledge base.
  3. [RELEVANCE GRADING] 1/4 relevant. -> Generating.
  4. [GENERATE] Response produced (attempt 1). 91 chars. Source: knowledge base.
  5. [HALLUCINATION CHECK] GROUNDED - response is faithful. Attempt 1/3.
```

**Final Response:**

> The prerequisites for the Machine Learning course (CS-302) are **CS-301** and **MATH-202**.

---

## Test Case 3: Retrieval IS needed but documents are irrelevant -> web search fallback

### Query

> What is the current stock price of XYZ National University's parent company?

### Expected Behavior

**Expected pipeline path:** `retrieve_docs -> grade_docs -> web_search -> generate_response -> hallucination_check -> END`

The query is about a stock price — something that cannot be in the university catalog. The
agent should retrieve documents (university-related query pattern), grade them all as
irrelevant, trigger the web search fallback, and generate an answer from web results.

### Actual Behavior

**Status:** ✅ Completed successfully
  - Retrieval needed: `True`
  - Web search used: `True`
  - Chunks retrieved: `4`
  - Chunks graded relevant: `0`
  - Hallucination detected: `False`
  - Retry count: `0`

**Actual pipeline path:**
> `decide_retrieval[retrieve] -> retrieve_docs(4 docs) -> grade_docs(0 relevant) -> web_search -> generate_response -> hallucination_check -> END`

**Decision Trace:**

```
  1. [ADAPTIVE RETRIEVAL] 'What is the current stock price of XYZ National University's parent company?' -> RETRIEVE from KB
  2. [RETRIEVAL] Retrieved 4 chunks from knowledge base.
  3. [RELEVANCE GRADING] 0/4 relevant. -> All irrelevant, triggering web search.
  4. [WEB SEARCH FALLBACK] Fetched 21 chars of web results.
  5. [GENERATE] Response produced (attempt 1). 173 chars. Source: web.
  6. [HALLUCINATION CHECK] GROUNDED - response is faithful. Attempt 1/3.
```

**Final Response:**

> I don't have enough information to answer your question. The context provided does not include any details about XYZ National University's parent company or its stock price.

---

## Test Case 4: Hallucination check fails -> agent regenerates

### Query

> List every single grade, GPA threshold, credit hours, attendance rule, fee structure, scholarship criteria, and exam schedule for all 27 courses offered by all three departments simultaneously.

### Expected Behavior

**Expected pipeline path:** `retrieve_docs -> grade_docs -> generate_response -> hallucination_check [hallucinated] -> generate_response (retry) -> hallucination_check -> END or disclaimer`

The query asks for extremely specific and comprehensive details that span all departments
simultaneously. The initial generation is likely to include fabricated details (exact
numbers, made-up thresholds) not explicitly supported by the retrieved chunks. The
hallucination check should catch this and trigger at least one retry. On retry the model
is prompted more strictly, and if it still fails after MAX_RETRIES attempts, a disclaimer
is issued.

### Actual Behavior

**Status:** ✅ Completed successfully
  - Retrieval needed: `True`
  - Web search used: `True`
  - Chunks retrieved: `4`
  - Chunks graded relevant: `0`
  - Hallucination detected: `False`
  - Retry count: `1`

**Actual pipeline path:**
> `decide_retrieval[retrieve] -> retrieve_docs(4 docs) -> grade_docs(0 relevant) -> web_search -> generate_response -> hallucination_check -> END`

**Decision Trace:**

```
  1. [ADAPTIVE RETRIEVAL] 'List every single grade, GPA threshold, credit hours, attendance rule, fee structure, scholarship criteria, and exam schedule for all 27 courses offered by all three departments simultaneously.' -> RETRIEVE from KB
  2. [RETRIEVAL] Retrieved 4 chunks from knowledge base.
  3. [RELEVANCE GRADING] 0/4 relevant. -> All irrelevant, triggering web search.
  4. [WEB SEARCH FALLBACK] Fetched 1742 chars of web results.
  5. [GENERATE] Response produced (attempt 1). 340 chars. Source: web.
  6. [HALLUCINATION CHECK] HALLUCINATION DETECTED - will retry. Attempt 1/3.
  7. [GENERATE] Response produced (RETRY 1). 340 chars. Source: web.
  8. [HALLUCINATION CHECK] GROUNDED - response is faithful. Attempt 2/3.
```

**Final Response:**

> The context provided does not contain any information about grades, GPA thresholds, credit hours, attendance rules, fee structures, scholarship criteria, or exam schedules for any courses at XYZ National University. It only includes details about Yankee Stadium in New York.

Therefore, I cannot answer your question with the given context.

---

## Test Case 5: Creative test — multi-department faculty query with grounding verification

### Query

> Which faculty members teach both CS and EE courses, and what are their office locations?

### Expected Behavior

**Expected pipeline path:** `retrieve_docs -> grade_docs -> generate_response -> hallucination_check -> END`

This is a cross-department faculty query. The agent should retrieve from
Faculty_Directory.pdf and possibly the department catalogs, grade only the faculty-
directory chunks as relevant, generate an answer listing faculty who bridge both
departments, and verify the response is grounded in the faculty directory data before
delivering it.

### Actual Behavior

**Status:** ✅ Completed successfully
  - Retrieval needed: `True`
  - Web search used: `True`
  - Chunks retrieved: `4`
  - Chunks graded relevant: `0`
  - Hallucination detected: `False`
  - Retry count: `1`

**Actual pipeline path:**
> `decide_retrieval[retrieve] -> retrieve_docs(4 docs) -> grade_docs(0 relevant) -> web_search -> generate_response -> hallucination_check -> END`

**Decision Trace:**

```
  1. [ADAPTIVE RETRIEVAL] 'Which faculty members teach both CS and EE courses, and what are their office locations?' -> RETRIEVE from KB
  2. [RETRIEVAL] Retrieved 4 chunks from knowledge base.
  3. [RELEVANCE GRADING] 0/4 relevant. -> All irrelevant, triggering web search.
  4. [WEB SEARCH FALLBACK] Fetched 21 chars of web results.
  5. [GENERATE] Response produced (attempt 1). 222 chars. Source: web.
  6. [HALLUCINATION CHECK] HALLUCINATION DETECTED - will retry. Attempt 1/3.
  7. [GENERATE] Response produced (RETRY 1). 247 chars. Source: web.
  8. [HALLUCINATION CHECK] GROUNDED - response is faithful. Attempt 2/3.
```

**Final Response:**

> The context does not provide any information about faculty members who teach both CS (Computer Science) and EE (Electrical Engineering) courses or their office locations. Therefore, I cannot answer your question based on the available information.

---

## Summary Table

| TC | Scenario | Retrieved | Web Search | Retries | Status |
|----|----------|-----------|------------|---------|--------|
| 1 | A query where retrieval is NOT needed… | No | No | 0 | ✅ |
| 2 | A query where retrieval IS needed and do… | Yes | No | 0 | ✅ |
| 3 | Retrieval IS needed but documents are ir… | Yes | Yes | 0 | ✅ |
| 4 | Hallucination check fails -> agent regen… | Yes | Yes | 1 | ✅ |
| 5 | Creative test — multi-department faculty… | Yes | Yes | 1 | ✅ |

---

*Report generated by `run_tests.py` — AI407L Spring 2026 Final Exam Part B*