"""
main.py — FastAPI application wrapping a LangGraph RAG pipeline for
Zepto customer-support queries.

Environment variables
---------------------
MOCK_LLM   "1" (default) → keyword heuristic + canned responses (graded baseline)
           "0"           → real LLM calls via LangChain / OpenAI (optional extension)
OPENAI_API_KEY  required only when MOCK_LLM=0
"""

from __future__ import annotations

import json
import os
from typing import Any

import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict

from prompt_template import build_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOCK_LLM: bool = os.environ.get("MOCK_LLM", "1") != "0"
COLLECTION_NAME = "zepto_policies"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

# ---------------------------------------------------------------------------
# Pydantic I/O models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    query: str
    intent: str                    # "policy_question" | "general_question"
    context: list[dict[str, Any]]  # retrieved chunks [{id, document}]
    response: dict[str, Any]       # serialised AnswerResponse


# ---------------------------------------------------------------------------
# Lazy-loaded shared resources
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None
_chroma_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


# ---------------------------------------------------------------------------
# Optional real-LLM helper (MOCK_LLM=0 only)
# ---------------------------------------------------------------------------


def _call_llm_with_retry(
    query: str,
    context: list[dict[str, Any]],
    max_retries: int = 3,
) -> AnswerResponse:
    """Prompt the real LLM and retry up to max_retries times on schema failure."""
    from langchain_openai import ChatOpenAI  # imported lazily — needs OPENAI_API_KEY

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = build_prompt(query, context)

    last_error: str = ""
    for attempt in range(max_retries):
        try:
            raw = llm.invoke(prompt).content.strip()
            # Strip optional markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return AnswerResponse(**data)
        except (json.JSONDecodeError, ValidationError, KeyError) as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                prompt += (
                    f"\n\nYour previous response failed validation ({exc}). "
                    "Please correct it and respond with valid JSON exactly matching "
                    "the required schema."
                )

    # All retries exhausted
    return AnswerResponse(
        answer=f"Error: could not produce a valid structured response after {max_retries} attempts. Last error: {last_error}",
        sources=[c["id"] for c in context],
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------


def classify_intent(state: GraphState) -> GraphState:
    """
    Node 1 — classify_intent

    Mock mode  : keyword heuristic, no LLM call.
    Real mode  : delegates to LLM (optional extension).
    """
    query_lower = state["query"].lower()

    if MOCK_LLM:
        intent = (
            "policy_question"
            if any(kw in query_lower for kw in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            classification_prompt = (
                "Classify the following customer query as exactly one of:\n"
                "  policy_question  — requires looking up a Zepto policy\n"
                "  general_question — does not require a policy lookup\n\n"
                f'Query: "{state["query"]}"\n\n'
                "Respond with only the label, nothing else."
            )
            label = llm.invoke(classification_prompt).content.strip().lower()
            intent = label if label in ("policy_question", "general_question") else "general_question"
        except Exception:
            # Fall back to keyword heuristic on any LLM failure
            intent = (
                "policy_question"
                if any(kw in query_lower for kw in POLICY_KEYWORDS)
                else "general_question"
            )

    return {**state, "intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    """
    Node 2 — retrieve_and_answer  (handles policy_question intents)

    Retrieval always runs for real (embedding + ChromaDB, no API key needed).
    Answer generation branches on MOCK_LLM:
      Mock  : canned "Based on the retrieved context: <top-chunk snippet>"
      Real  : structured prompt → LLM with retry-on-schema-failure
    """
    query = state["query"]
    model = get_embedding_model()
    collection = get_chroma_collection()

    # --- Real retrieval (both modes) ---
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"],
    )
    context = [
        {"id": results["ids"][0][i], "document": results["documents"][0][i]}
        for i in range(len(results["ids"][0]))
    ]

    # --- Answer generation ---
    if MOCK_LLM:
        top_snippet = (context[0]["document"][:200] if context else "")
        response = AnswerResponse(
            answer=f"Based on the retrieved context: {top_snippet}",
            sources=[c["id"] for c in context],
            confidence=1.0,
        )
    else:
        response = _call_llm_with_retry(query, context)

    return {**state, "context": context, "response": response.model_dump()}


def direct_answer(state: GraphState) -> GraphState:
    """
    Node 3 — direct_answer  (handles general_question intents)

    Mock  : fixed canned string, no LLM call.
    Real  : delegates to LLM directly (optional extension).
    """
    if MOCK_LLM:
        response = AnswerResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
    else:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            raw = llm.invoke(state["query"]).content
            response = AnswerResponse(answer=raw, sources=[], confidence=0.8)
        except Exception:
            response = AnswerResponse(
                answer="I can only answer questions about Zepto policies right now.",
                sources=[],
                confidence=1.0,
            )

    return {**state, "response": response.model_dump()}


# ---------------------------------------------------------------------------
# Conditional routing (does NOT depend on MOCK_LLM)
# ---------------------------------------------------------------------------


def route_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Build the LangGraph StateGraph
# ---------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Zepto Policy RAG API",
    description="LangGraph-based RAG pipeline for Zepto customer-support queries.",
    version="1.0.0",
)

rag_graph = build_graph()


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QueryRequest):
    """Accept a customer query and return a structured policy answer."""
    try:
        initial_state: GraphState = {
            "query": request.query,
            "intent": "",
            "context": [],
            "response": {},
        }
        result = rag_graph.invoke(initial_state)
        return AnswerResponse(**result["response"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health():
    return {"status": "ok", "mock_llm": MOCK_LLM}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
