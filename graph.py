"""
graph.py — Self-RAG LangGraph StateGraph
University Course Advisory Agent — XYZ National University
Uses Mistral LLM for all decision-making, grading, and generation.

Pipeline:
    START -> decide_retrieval
               |- [direct]   -> direct_answer -> END
               |- [retrieve] -> retrieve_docs -> grade_docs
                                    |- [relevant] -> generate_response
                                    |- [all irrelevant] -> web_search -> generate_response
                                                             -> hallucination_check
                                                                  |- [grounded]    -> END
                                                                  |- [retry < max] -> generate_response
                                                                  |- [max retries] -> disclaimer -> END
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI  # Mistral LLM for all decision nodes
from langgraph.graph import END, START, StateGraph

from tools import (
    check_hallucination,
    grade_document_relevance,
    retrieve_from_knowledge_base,
    web_search_fallback,
)

load_dotenv()

MAX_RETRIES: int = 2


class SelfRAGState(TypedDict):
    query: str
    needs_retrieval: bool
    retrieved_docs: list
    relevant_docs: list
    context: str
    response: str
    hallucination_detected: bool
    retry_count: int
    decision_trace: list
    web_search_used: bool


def _llm() -> ChatMistralAI:
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.1,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────

def decide_retrieval_node(state: SelfRAGState) -> dict:
    """Adaptive Retrieval: decide whether to query the KB or answer directly."""
    query = state["query"]
    prompt = (
        "You are a retrieval decision router for a University Course Advisory system.\n\n"
        "Classify whether the student query requires searching the university knowledge base "
        "(courses, prerequisites, credit hours, grades, fees, policies, faculty) or can be "
        "answered directly from general knowledge.\n\n"
        "Rules:\n"
        "- Greetings, small talk, generic knowledge -> reply 'direct'\n"
        "- Any XYZ National University specific question -> reply 'retrieve'\n\n"
        "Examples:\n"
        "  'Hello!' -> direct\n"
        "  'What does GPA stand for?' -> direct\n"
        "  'What are CS301 prerequisites?' -> retrieve\n"
        "  'How much is the semester fee?' -> retrieve\n\n"
        f"Student Query: {query}\n\n"
        "Decision (reply ONLY 'direct' or 'retrieve'):"
    )
    result = _llm().invoke(prompt)
    needs_retrieval = "retrieve" in result.content.strip().lower()

    msg = (
        f"[ADAPTIVE RETRIEVAL] '{query}' -> "
        f"{'RETRIEVE from KB' if needs_retrieval else 'DIRECT answer (no retrieval)'}"
    )
    print(f"\n{msg}")

    return {
        "needs_retrieval": needs_retrieval,
        "decision_trace": state.get("decision_trace", []) + [msg],
        "retrieved_docs": [],
        "relevant_docs": [],
        "context": "",
        "response": "",
        "hallucination_detected": False,
        "retry_count": 0,
        "web_search_used": False,
    }


def retrieve_docs_node(state: SelfRAGState) -> dict:
    """Retrieve top-k chunks from ChromaDB."""
    query = state["query"]
    print(f"[RETRIEVAL] Searching KB for: '{query}'")

    docs = retrieve_from_knowledge_base.invoke({"query": query, "k": 4})
    msg = f"[RETRIEVAL] Retrieved {len(docs)} chunks from knowledge base."
    print(msg)

    return {
        "retrieved_docs": docs,
        "decision_trace": state.get("decision_trace", []) + [msg],
    }


def grade_docs_node(state: SelfRAGState) -> dict:
    """Relevance Grading: assess each chunk individually; discard irrelevant ones."""
    query = state["query"]
    retrieved = state.get("retrieved_docs", [])
    relevant = []

    print(f"[GRADING] Evaluating {len(retrieved)} chunks...")

    for i, doc in enumerate(retrieved):
        grade = grade_document_relevance.invoke(
            {"document_content": doc["content"], "query": query}
        )
        icon = "OK" if grade == "relevant" else "XX"
        src = doc["metadata"].get("source_file", "unknown")
        dept = doc["metadata"].get("department", "")
        print(f"  Chunk {i+1}: [{icon}] {grade.upper()} - [{dept}] {src}")
        if grade == "relevant":
            relevant.append(doc)

    suffix = "-> Generating." if relevant else "-> All irrelevant, triggering web search."
    msg = f"[RELEVANCE GRADING] {len(relevant)}/{len(retrieved)} relevant. {suffix}"
    print(msg)

    return {
        "relevant_docs": relevant,
        "decision_trace": state.get("decision_trace", []) + [msg],
    }


def web_search_node(state: SelfRAGState) -> dict:
    """Web Search Fallback: triggered when KB has no relevant documents."""
    query = state["query"]
    print(f"[WEB SEARCH] Falling back to web search for: '{query}'")

    results = web_search_fallback.invoke({"query": query})
    context = f"[Web Search Results - Query: '{query}']\n\n{results}"

    msg = f"[WEB SEARCH FALLBACK] Fetched {len(results)} chars of web results."
    print(msg)

    return {
        "context": context,
        "web_search_used": True,
        "decision_trace": state.get("decision_trace", []) + [msg],
    }


def generate_response_node(state: SelfRAGState) -> dict:
    """Generate an answer from relevant KB chunks or web search results."""
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    web_used = state.get("web_search_used", False)

    if not web_used:
        parts = []
        for doc in state.get("relevant_docs", []):
            dept = doc["metadata"].get("department", "")
            dtype = doc["metadata"].get("doc_type", "")
            parts.append(f"[Source: {dept} - {dtype}]\n{doc['content']}")
        context = "\n\n---\n\n".join(parts)
    else:
        context = state.get("context", "")

    label = f"attempt {retry_count + 1}" if retry_count == 0 else f"RETRY {retry_count}"
    print(f"[GENERATE] Generating response ({label})...")

    strict_note = (
        "IMPORTANT: Only use facts explicitly stated in the context. "
        "Do NOT invent any course codes, credit hours, names, or numbers.\n"
        if retry_count > 0
        else ""
    )

    prompt = (
        "You are a helpful University Course Advisory Agent for XYZ National University.\n"
        "Answer the student's question using ONLY the information in the context below.\n"
        "If the context lacks enough information, say so clearly — do NOT guess.\n"
        f"{strict_note}\n"
        f"Context:\n{context}\n\n"
        f"Student Question: {query}\n\n"
        "Answer:"
    )

    result = _llm().invoke(prompt)
    response = result.content.strip()

    msg = (
        f"[GENERATE] Response produced ({label}). "
        f"{len(response)} chars. Source: {'web' if web_used else 'knowledge base'}."
    )
    print(msg)

    return {
        "context": context,
        "response": response,
        "decision_trace": state.get("decision_trace", []) + [msg],
    }


def direct_answer_node(state: SelfRAGState) -> dict:
    """Answer conversational or general-knowledge queries without retrieval."""
    query = state["query"]
    print(f"[DIRECT] Answering from general knowledge...")

    prompt = (
        "You are a helpful University Course Advisory Agent for XYZ National University.\n"
        "The student asked a general or conversational question. Answer naturally.\n\n"
        f"Student Question: {query}\n\nAnswer:"
    )

    result = _llm().invoke(prompt)
    msg = "[DIRECT ANSWER] Responded from general knowledge. No KB retrieval performed."
    print(msg)

    return {
        "context": "General knowledge - no retrieval.",
        "response": result.content.strip(),
        "decision_trace": state.get("decision_trace", []) + [msg],
    }


def hallucination_check_node(state: SelfRAGState) -> dict:
    """Verify the generated response is grounded; increment retry_count if not."""
    response = state.get("response", "")
    context = state.get("context", "")
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    print(f"[HALLUCINATION CHECK] Verifying response (attempt {retry_count + 1})...")

    grade = check_hallucination.invoke(
        {"response": response, "context": context, "query": query}
    )
    hallucinated = grade == "hallucinated"
    new_retry = retry_count + 1 if hallucinated else retry_count

    status = "HALLUCINATION DETECTED - will retry" if hallucinated else "GROUNDED - response is faithful"
    msg = (
        f"[HALLUCINATION CHECK] {status}. "
        f"Attempt {retry_count + 1}/{MAX_RETRIES + 1}."
    )
    print(msg)

    return {
        "hallucination_detected": hallucinated,
        "retry_count": new_retry,
        "decision_trace": state.get("decision_trace", []) + [msg],
    }


def disclaimer_node(state: SelfRAGState) -> dict:
    """Issue a disclaimer when max retries are exhausted."""
    msg = f"[DISCLAIMER] Max retries ({MAX_RETRIES}) exhausted. Issuing disclaimer."
    print(f"\n{msg}")
    return {
        "response": (
            "I was unable to provide a fully verified answer after multiple attempts. "
            "The information could not be confirmed against the official university documents. "
            "Please contact XYZ National University admissions directly for accurate information."
        ),
        "decision_trace": state.get("decision_trace", []) + [msg],
    }


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_retrieval_decision(state: SelfRAGState) -> str:
    return "retrieve_docs" if state["needs_retrieval"] else "direct_answer"


def route_after_grading(state: SelfRAGState) -> str:
    return "generate_response" if state.get("relevant_docs") else "web_search"


def route_after_hallucination_check(state: SelfRAGState) -> str:
    if not state.get("hallucination_detected", False):
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "disclaimer"
    return "generate_response"


# ── Graph construction ────────────────────────────────────────────────────────

def build_self_rag_graph():
    graph = StateGraph(SelfRAGState)

    graph.add_node("decide_retrieval", decide_retrieval_node)
    graph.add_node("retrieve_docs", retrieve_docs_node)
    graph.add_node("grade_docs", grade_docs_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("hallucination_check", hallucination_check_node)
    graph.add_node("disclaimer", disclaimer_node)

    graph.add_edge(START, "decide_retrieval")

    graph.add_conditional_edges(
        "decide_retrieval",
        route_after_retrieval_decision,
        {"retrieve_docs": "retrieve_docs", "direct_answer": "direct_answer"},
    )

    graph.add_edge("retrieve_docs", "grade_docs")

    graph.add_conditional_edges(
        "grade_docs",
        route_after_grading,
        {"generate_response": "generate_response", "web_search": "web_search"},
    )

    graph.add_edge("web_search", "generate_response")
    graph.add_edge("generate_response", "hallucination_check")

    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {"end": END, "generate_response": "generate_response", "disclaimer": "disclaimer"},
    )

    graph.add_edge("direct_answer", END)
    graph.add_edge("disclaimer", END)

    return graph.compile()


if __name__ == "__main__":
    agent = build_self_rag_graph()
    print("[OK] Self-RAG graph compiled successfully.")
