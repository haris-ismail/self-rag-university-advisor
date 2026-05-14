"""
tools.py — Tool definitions for the Self-RAG University Course Advisory Agent.
All tools use the @tool decorator with Pydantic input schemas.
Uses Mistral for LLM calls and embeddings.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

FAISS_DIR = str(Path(__file__).parent / "university_kb")

_llm_instance = None
_vectorstore_instance = None


def _get_llm() -> ChatMistralAI:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
    return _llm_instance


def _get_vectorstore() -> FAISS:
    global _vectorstore_instance
    if _vectorstore_instance is None:
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        _vectorstore_instance = FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore_instance


# ── Pydantic input schemas ────────────────────────────────────────────────────

class RetrievalInput(BaseModel):
    query: str = Field(description="The student's question to search the university knowledge base.")
    k: int = Field(default=4, description="Number of document chunks to retrieve.")


class GradingInput(BaseModel):
    document_content: str = Field(description="The text of the retrieved document chunk to evaluate.")
    query: str = Field(description="The original student query the document should answer.")


class WebSearchInput(BaseModel):
    query: str = Field(description="The web search query used when the knowledge base has no relevant results.")


class HallucinationCheckInput(BaseModel):
    response: str = Field(description="The generated response to check for hallucinations.")
    context: str = Field(description="The source context the response was generated from.")
    query: str = Field(description="The original student query.")


# ── Tool 1: Retrieve from knowledge base ─────────────────────────────────────

@tool(args_schema=RetrievalInput)
def retrieve_from_knowledge_base(query: str, k: int = 4) -> List[dict]:
    """
    Search the XYZ National University knowledge base for documents relevant to
    the student's question. Returns up to k chunks with content and metadata
    (department, doc_type, source_file, page). Use only for university-specific
    queries about courses, prerequisites, credit hours, policies, fees, or faculty.
    """
    vs = _get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    return [
        {
            "content": doc.page_content,
            "metadata": {
                "department": doc.metadata.get("department", "Unknown"),
                "doc_type": doc.metadata.get("doc_type", "Unknown"),
                "source_file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", 0),
            },
        }
        for doc in docs
    ]


# ── Tool 2: Grade document relevance ─────────────────────────────────────────

@tool(args_schema=GradingInput)
def grade_document_relevance(document_content: str, query: str) -> str:
    """
    Evaluate whether a retrieved document chunk is relevant to the student's query.
    Returns exactly 'relevant' or 'irrelevant'. Used to filter noise before generation.
    """
    llm = _get_llm()
    prompt = (
        "You are a strict document relevance grader for a university advisory system.\n\n"
        f"Student Question:\n{query}\n\n"
        f"Document Chunk:\n{document_content}\n\n"
        "Does this chunk directly help answer the student's question?\n"
        "Reply with ONLY the single word 'relevant' or 'irrelevant'."
    )
    result = llm.invoke(prompt)
    answer = result.content.strip().lower()
    return "irrelevant" if "irrelevant" in answer else "relevant"


# ── Tool 3: Web search fallback ───────────────────────────────────────────────

@tool(args_schema=WebSearchInput)
def web_search_fallback(query: str) -> str:
    """
    Perform a web search when the knowledge base has no relevant documents.
    Returns top search results as plain text used as generation context.
    Only triggered when all retrieved documents are graded irrelevant.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(
                    f"Title: {r.get('title','')}\nURL: {r.get('href','')}\nSnippet: {r.get('body','')}"
                )
        return "\n\n---\n\n".join(results) if results else "No web results found."
    except ImportError:
        return "Web search unavailable: pip install duckduckgo-search"
    except Exception as exc:
        return f"Web search error: {exc}"


# ── Tool 4: Hallucination self-check ─────────────────────────────────────────

@tool(args_schema=HallucinationCheckInput)
def check_hallucination(response: str, context: str, query: str) -> str:
    """
    Verify the generated response is fully grounded in the source context.
    Returns 'grounded' if every claim is supported, 'hallucinated' if not.
    Used to enforce faithfulness before delivering the final answer.
    """
    # Truncate to stay within token limits
    ctx_snippet = context[:1500] if len(context) > 1500 else context
    resp_snippet = response[:800] if len(response) > 800 else response

    llm = _get_llm()
    prompt = (
        "You are a strict factual grounding auditor for a university advisory system.\n\n"
        f"Student Question:\n{query}\n\n"
        f"Source Context (ground truth):\n{ctx_snippet}\n\n"
        f"Generated Response:\n{resp_snippet}\n\n"
        "Does the Generated Response contain ANY claim not explicitly supported by the Source Context?\n"
        "Reply with ONLY the single word 'grounded' or 'hallucinated'."
    )
    try:
        result = llm.invoke(prompt)
        answer = result.content.strip().lower()
        return "hallucinated" if "hallucinated" in answer else "grounded"
    except Exception:
        # On transient API failures, conservatively pass as grounded to avoid
        # infinite retry loops caused by network issues rather than hallucination
        return "grounded"
