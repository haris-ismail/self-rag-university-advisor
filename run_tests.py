"""
run_tests.py — Automated test runner for the Self-RAG University Advisory Agent.
Executes all 5 required test scenarios, captures traces, and writes
evaluation_results.md automatically.

Test Scenarios (per exam rubric):
    TC1 — No retrieval needed (greeting / general knowledge)
    TC2 — Retrieval needed, documents are relevant
    TC3 — Retrieval needed, documents are irrelevant -> web search fallback
    TC4 — Hallucination detected -> agent regenerates
    TC5 — Creative / edge-case query

Usage:
    python run_tests.py           # run all 5 tests and write evaluation_results.md
    python run_tests.py --dry     # just print expected behaviour without running the agent
"""

import argparse
import sys
import textwrap
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# -- Test case definitions -----------------------------------------------------

TEST_CASES = [
    {
        "id": 1,
        "scenario": "A query where retrieval is NOT needed",
        "query": "Hello! Can you tell me what GPA stands for?",
        "expected_path": "direct_answer",
        "expected_reasoning": (
            "The query is a greeting combined with a general-knowledge question. "
            "'GPA' (Grade Point Average) is common knowledge that does not require "
            "searching the university knowledge base. The agent should classify this "
            "as 'direct' and answer immediately without any retrieval."
        ),
    },
    {
        "id": 2,
        "scenario": "A query where retrieval IS needed and documents are relevant",
        "query": "What are the prerequisites for the Machine Learning course in the CS department?",
        "expected_path": "retrieve_docs -> grade_docs -> generate_response -> hallucination_check -> END",
        "expected_reasoning": (
            "This is a specific university question about a CS course's prerequisites. "
            "The agent should classify this as 'retrieve', fetch chunks from "
            "CS_Department_Catalog.pdf, grade them as relevant, generate an answer "
            "grounded in those chunks, pass the hallucination check, and deliver the answer."
        ),
    },
    {
        "id": 3,
        "scenario": "Retrieval IS needed but documents are irrelevant -> web search fallback",
        "query": "What is the current stock price of XYZ National University's parent company?",
        "expected_path": "retrieve_docs -> grade_docs -> web_search -> generate_response -> hallucination_check -> END",
        "expected_reasoning": (
            "The query is about a stock price — something that cannot be in the university "
            "catalog. The agent should retrieve documents (university-related query pattern), "
            "grade them all as irrelevant, trigger the web search fallback, and generate "
            "an answer from web results."
        ),
    },
    {
        "id": 4,
        "scenario": "Hallucination check fails -> agent regenerates",
        "query": (
            "List every single grade, GPA threshold, credit hours, attendance rule, "
            "fee structure, scholarship criteria, and exam schedule for all 27 courses "
            "offered by all three departments simultaneously."
        ),
        "expected_path": (
            "retrieve_docs -> grade_docs -> generate_response -> hallucination_check "
            "[hallucinated] -> generate_response (retry) -> hallucination_check -> END or disclaimer"
        ),
        "expected_reasoning": (
            "The query asks for extremely specific and comprehensive details that span "
            "all departments simultaneously. The initial generation is likely to include "
            "fabricated details (exact numbers, made-up thresholds) not explicitly supported "
            "by the retrieved chunks. The hallucination check should catch this and trigger "
            "at least one retry. On retry the model is prompted more strictly, and if it "
            "still fails after MAX_RETRIES attempts, a disclaimer is issued."
        ),
    },
    {
        "id": 5,
        "scenario": "Creative test — multi-department faculty query with grounding verification",
        "query": "Which faculty members teach both CS and EE courses, and what are their office locations?",
        "expected_path": "retrieve_docs -> grade_docs -> generate_response -> hallucination_check -> END",
        "expected_reasoning": (
            "This is a cross-department faculty query. The agent should retrieve from "
            "Faculty_Directory.pdf and possibly the department catalogs, grade only the "
            "faculty-directory chunks as relevant, generate an answer listing faculty "
            "who bridge both departments, and verify the response is grounded in the "
            "faculty directory data before delivering it."
        ),
    },
]


# -- Runner --------------------------------------------------------------------

def run_all_tests(agent) -> list:
    results = []
    for tc in TEST_CASES:
        print(f"\n{'=' * 70}")
        print(f"TEST CASE {tc['id']}: {tc['scenario']}")
        print(f"Query: {tc['query']!r}")
        print(f"Expected path: {tc['expected_path']}")
        print("-" * 70)

        state = {
            "query": tc["query"],
            "needs_retrieval": False,
            "retrieved_docs": [],
            "relevant_docs": [],
            "context": "",
            "response": "",
            "hallucination_detected": False,
            "retry_count": 0,
            "decision_trace": [],
            "web_search_used": False,
        }

        try:
            final_state = agent.invoke(state)
            results.append(
                {
                    "tc": tc,
                    "final_state": final_state,
                    "error": None,
                }
            )
        except Exception as exc:
            print(f"[ERROR] Test case {tc['id']} failed with exception: {exc}")
            results.append({"tc": tc, "final_state": None, "error": str(exc)})

        print(f"\n[TC{tc['id']} COMPLETE]")

    return results


# -- Report generation ---------------------------------------------------------

def _fmt_trace(trace: list) -> str:
    if not trace:
        return "  *(no trace recorded)*"
    return "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(trace))


def _wrap(text: str, width: int = 90) -> str:
    return "\n".join(
        textwrap.fill(line, width) if line.strip() else ""
        for line in text.splitlines()
    )


def write_evaluation_results(results: list, output_path: Path) -> None:
    lines = []
    lines.append("# Evaluation Results — Self-RAG University Course Advisory Agent")
    lines.append("")
    lines.append(f"**Course:** AI407L Capstone Lab — Spring 2026  ")
    lines.append(f"**Part:** B — Self-RAG Agent  ")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Model:** mistral-small-latest")
    lines.append(f"**Vector Store:** FAISS (university_kb/)  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "This document records the execution of all 5 required Self-RAG test scenarios. "
        "Each section shows the query, the expected pipeline path, the actual decision trace "
        "captured during execution, and the agent's final response."
    )
    lines.append("")
    lines.append("### Self-RAG Decision Paths")
    lines.append("")
    lines.append("```")
    lines.append("START -> decide_retrieval")
    lines.append("          +-[direct]---> direct_answer -> END")
    lines.append("          +-[retrieve]-> retrieve_docs -> grade_docs")
    lines.append("                                          +-[relevant docs]---> generate_response")
    lines.append("                                          +-[no relevant]-----> web_search -> generate_response")
    lines.append("                                                                               ↓")
    lines.append("                                                                   hallucination_check")
    lines.append("                                                                       +-[grounded]-----> END")
    lines.append("                                                                       +-[retry]--------> generate_response")
    lines.append("                                                                       +-[max retries]--> disclaimer -> END")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")

    for item in results:
        tc = item["tc"]
        final = item["final_state"]
        err = item["error"]

        lines.append(f"## Test Case {tc['id']}: {tc['scenario']}")
        lines.append("")
        lines.append(f"### Query")
        lines.append("")
        lines.append(f"> {tc['query']}")
        lines.append("")
        lines.append(f"### Expected Behavior")
        lines.append("")
        lines.append(f"**Expected pipeline path:** `{tc['expected_path']}`")
        lines.append("")
        lines.append(_wrap(tc["expected_reasoning"]))
        lines.append("")
        lines.append(f"### Actual Behavior")
        lines.append("")

        if err:
            lines.append(f"**Status:** ❌ Error during execution")
            lines.append("")
            lines.append(f"```")
            lines.append(f"ERROR: {err}")
            lines.append(f"```")
        elif final:
            needs_ret = final.get("needs_retrieval", False)
            web_used = final.get("web_search_used", False)
            hallucination = final.get("hallucination_detected", False)
            retries = final.get("retry_count", 0)
            n_retrieved = len(final.get("retrieved_docs", []))
            n_relevant = len(final.get("relevant_docs", []))

            # Infer actual path
            if not needs_ret:
                actual_path = "decide_retrieval[direct] -> direct_answer -> END"
            elif web_used:
                actual_path = f"decide_retrieval[retrieve] -> retrieve_docs({n_retrieved} docs) -> grade_docs({n_relevant} relevant) -> web_search -> generate_response -> hallucination_check -> END"
            else:
                actual_path = f"decide_retrieval[retrieve] -> retrieve_docs({n_retrieved} docs) -> grade_docs({n_relevant} relevant) -> generate_response"
                if retries > 0:
                    actual_path += f" -> hallucination_check[hallucinated] × {retries}"
                actual_path += " -> hallucination_check[grounded/disclaimer] -> END"

            lines.append(f"**Status:** ✅ Completed successfully")
            lines.append(f"  - Retrieval needed: `{needs_ret}`")
            lines.append(f"  - Web search used: `{web_used}`")
            lines.append(f"  - Chunks retrieved: `{n_retrieved}`")
            lines.append(f"  - Chunks graded relevant: `{n_relevant}`")
            lines.append(f"  - Hallucination detected: `{hallucination}`")
            lines.append(f"  - Retry count: `{retries}`")
            lines.append("")
            lines.append(f"**Actual pipeline path:**")
            lines.append(f"> `{actual_path}`")
            lines.append("")
            lines.append("**Decision Trace:**")
            lines.append("")
            lines.append("```")
            lines.append(_fmt_trace(final.get("decision_trace", [])))
            lines.append("```")
            lines.append("")
            lines.append("**Final Response:**")
            lines.append("")
            lines.append(f"> {final.get('response', '(no response)')}")
        else:
            lines.append("**Status:** ❌ No result returned.")

        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Summary Table")
    lines.append("")
    lines.append("| TC | Scenario | Retrieved | Web Search | Retries | Status |")
    lines.append("|----|----------|-----------|------------|---------|--------|")
    for item in results:
        tc = item["tc"]
        final = item["final_state"]
        if item["error"]:
            lines.append(f"| {tc['id']} | {tc['scenario'][:40]}… | — | — | — | ❌ Error |")
        elif final:
            ret = "Yes" if final.get("needs_retrieval") else "No"
            web = "Yes" if final.get("web_search_used") else "No"
            retries = final.get("retry_count", 0)
            lines.append(f"| {tc['id']} | {tc['scenario'][:40]}… | {ret} | {web} | {retries} | ✅ |")
        else:
            lines.append(f"| {tc['id']} | {tc['scenario'][:40]}… | — | — | — | ❌ |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `run_tests.py` — AI407L Spring 2026 Final Exam Part B*")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[REPORT] evaluation_results.md written to {output_path}")


# -- Dry-run mode --------------------------------------------------------------

def dry_run() -> None:
    print("\n[DRY RUN] Test cases (no agent execution):\n")
    for tc in TEST_CASES:
        print(f"  TC{tc['id']}: {tc['scenario']}")
        print(f"    Query:         {tc['query'][:80]}{'...' if len(tc['query']) > 80 else ''}")
        print(f"    Expected path: {tc['expected_path']}")
        print()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all 5 Self-RAG test scenarios.")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Print test definitions without running the agent.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Build/rebuild the knowledge base before running tests.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force-rebuild the knowledge base.",
    )
    args = parser.parse_args()

    if args.dry:
        dry_run()
        return

    if args.ingest or args.rebuild:
        print("[STARTUP] Ingesting university documents...")
        from ingest import build_knowledge_base
        build_knowledge_base(force_rebuild=args.rebuild)
        print()

    print("[STARTUP] Compiling Self-RAG graph...", end=" ", flush=True)
    from graph import build_self_rag_graph
    agent = build_self_rag_graph()
    print("Ready.")

    results = run_all_tests(agent)

    output_path = Path(__file__).parent / "evaluation_results.md"
    write_evaluation_results(results, output_path)

    passed = sum(1 for r in results if r["error"] is None and r["final_state"] is not None)
    print(f"\n{'=' * 70}")
    print(f"TEST SUMMARY: {passed}/{len(TEST_CASES)} test cases completed successfully.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
