"""
app.py — Flask web server for the Self-RAG University Course Advisory Agent.
Serves the chat UI and exposes a /api/query endpoint.

Usage:
    python app.py
    Then open http://localhost:5000 in your browser.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

app = Flask(__name__)

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

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from graph import build_self_rag_graph
        _agent = build_self_rag_graph()
    return _agent


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    user_query = data.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    try:
        agent = _get_agent()
        state = {**INITIAL_STATE, "query": user_query}
        result = agent.invoke(state)

        retrieved = result.get("retrieved_docs", [])
        relevant = result.get("relevant_docs", [])

        return jsonify({
            "response": result.get("response", ""),
            "trace": result.get("decision_trace", []),
            "stats": {
                "needs_retrieval": result.get("needs_retrieval", False),
                "web_search_used": result.get("web_search_used", False),
                "retrieved_count": len(retrieved) if isinstance(retrieved, list) else 0,
                "relevant_count": len(relevant) if isinstance(relevant, list) else 0,
                "retry_count": result.get("retry_count", 0),
                "hallucination_detected": result.get("hallucination_detected", False),
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  XYZ National University — Course Advisory Agent")
    print("  Self-RAG Web Interface")
    print("=" * 60)
    print("[STARTUP] Compiling Self-RAG graph...", end=" ", flush=True)
    _get_agent()
    print("Ready.")
    print("[SERVER]  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
