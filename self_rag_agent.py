"""
self_rag_agent.py — Main interactive entry point for the Self-RAG University
Course Advisory Agent (XYZ National University).

Usage:
    python self_rag_agent.py              # interactive mode
    python self_rag_agent.py --ingest     # rebuild knowledge base then start
    python self_rag_agent.py --query "What are the prerequisites for CS301?"
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure the Part_B directory is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

BANNER = """
+======================================================================╗
|        XYZ National University — Course Advisory Agent (Self-RAG)   |
|               AI407L Final Exam — Part B                            |
+======================================================================+
|  Ask me about:                                                       |
|    • Course prerequisites, credit hours, and descriptions            |
|    • Departments: Computer Science · Electrical Engineering · BBA    |
|    • Grading policies, GPA rules, and academic calendar              |
|    • Tuition fees, withdrawal policies, and academic procedures      |
|    • Faculty names, offices, emails, and specializations             |
+======================================================================+
|  Commands:  'trace' — show last decision trace                       |
|             'quit'  — exit                                           |
+======================================================================╝
"""

INITIAL_STATE = {
    "query": "",
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


def run_query(agent, query: str) -> dict:
    """Run a single query through the Self-RAG graph and return the final state."""
    state = {**INITIAL_STATE, "query": query}
    return agent.invoke(state)


def print_trace(trace: list) -> None:
    if not trace:
        print("  (no trace available)")
        return
    print("\n" + "-" * 60)
    print("  DECISION TRACE")
    print("-" * 60)
    for step in trace:
        print(f"  {step}")
    print("-" * 60 + "\n")


def interactive_mode(agent) -> None:
    print(BANNER)
    last_trace: list = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "trace":
            print_trace(last_trace)
            continue

        print()  # spacing before pipeline output
        result = run_query(agent, user_input)
        last_trace = result.get("decision_trace", [])

        print(f"\n{'-' * 60}")
        print(f"Agent: {result['response']}")
        print(f"{'-' * 60}")
        print("(Type 'trace' to see how the agent reached this answer)\n")


def single_query_mode(agent, query: str) -> None:
    print(f"\nRunning query: {query!r}\n")
    result = run_query(agent, query)
    print(f"\n{'=' * 60}")
    print(f"Final Answer:\n{result['response']}")
    print(f"{'=' * 60}")
    print_trace(result.get("decision_trace", []))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-RAG University Course Advisory Agent"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Rebuild the knowledge base from PDFs before starting.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force-rebuild the knowledge base even if it already exists.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query and exit (non-interactive mode).",
    )
    args = parser.parse_args()

    # Optionally (re)build the knowledge base
    if args.ingest or args.rebuild:
        print("[STARTUP] Ingesting university documents into ChromaDB...")
        from ingest import build_knowledge_base
        build_knowledge_base(force_rebuild=args.rebuild)
        print()

    # Compile the Self-RAG graph
    print("[STARTUP] Compiling Self-RAG LangGraph...", end=" ", flush=True)
    from graph import build_self_rag_graph
    agent = build_self_rag_graph()
    print("Ready.\n")

    if args.query:
        single_query_mode(agent, args.query)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
