"""
Financial Market Intelligence Agent
-----------------------------------
LLM: Groq
RAG: FAISS + Hugging Face embeddings
Agent orchestration: LangGraph
"""

import os
import logging
from pathlib import Path
from typing import TypedDict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from src.rag_pipeline import load_pipeline, ask_question


# --------------------------------------------------
# 1. Environment and logging
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to your project's .env file."
    )

if not GROQ_MODEL:
    raise ValueError(
        "GROQ_MODEL is missing. Add a supported model ID to your .env file."
    )


# --------------------------------------------------
# 2. Initialize Groq
# --------------------------------------------------

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=1024,
    max_retries=2,
    timeout=60,
)


# --------------------------------------------------
# 3. Financial agent state
# --------------------------------------------------

class FinancialAgentState(TypedDict, total=False):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    retrieved_docs: list[str]
    latency: float
    validation_status: str
    retry_count: int
    agent_trace: list[str]


# --------------------------------------------------
# 4. Lazy-load the RAG pipeline
# --------------------------------------------------

_rag_chain = None
_agent = None


def get_rag_chain():
    """Load the financial RAG pipeline only when required."""
    global _rag_chain

    if _rag_chain is None:
        logger.info("Loading financial RAG pipeline...")
        _rag_chain = load_pipeline()
        logger.info("Financial RAG pipeline loaded.")

    return _rag_chain


# --------------------------------------------------
# 5. Planner node
# --------------------------------------------------

def planner_node(state: FinancialAgentState) -> dict:
    """Validate and prepare the user's financial question."""

    question = state.get("question", "").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    logger.info("Planner processing financial question.")

    return {
        "question": question,
        "retry_count": state.get("retry_count", 0),
        "agent_trace": state.get("agent_trace", []) + ["planner"],
    }


# --------------------------------------------------
# 6. Financial RAG retrieval node
# --------------------------------------------------

def retrieval_node(state: FinancialAgentState) -> dict:
    """
    Retrieve relevant financial evidence and generate an answer
    through the configured RAG pipeline.
    """

    logger.info("Retrieving financial evidence.")

    result = ask_question(
        get_rag_chain(),
        state["question"],
    )

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "retrieved_docs": result.get("retrieved_docs", []),
        "latency": result.get("latency", 0.0),
        "agent_trace": state.get("agent_trace", [])
        + ["financial_rag"],
    }


# --------------------------------------------------
# 7. Answer validation node
# --------------------------------------------------

def validator_node(state: FinancialAgentState) -> dict:
    """
    Perform basic structural validation.

    This checks whether an answer and retrieved evidence exist.
    It does not prove that every statement is factually grounded.
    """

    answer = state.get("answer", "").strip()
    documents = state.get("retrieved_docs", [])

    if not answer:
        status = "failed"

    elif not documents:
        status = "insufficient_evidence"

    else:
        status = "evidence_available"

    logger.info("Validation status: %s", status)

    return {
        "validation_status": status,
        "agent_trace": state.get("agent_trace", [])
        + ["validator"],
    }


# --------------------------------------------------
# 8. Conditional retry routing
# --------------------------------------------------

def route_after_validation(state: FinancialAgentState) -> str:
    """Retry once if generation returned an empty answer."""

    if (
        state.get("validation_status") == "failed"
        and state.get("retry_count", 0) < 1
    ):
        return "retry"

    return "finish"


def retry_node(state: FinancialAgentState) -> dict:
    """Record a retry before running retrieval again."""

    logger.warning("Retrying financial question.")

    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "agent_trace": state.get("agent_trace", []) + ["retry"],
    }


# --------------------------------------------------
# 9. Build the LangGraph workflow
# --------------------------------------------------

def build_agent():
    """Construct and compile the financial agent graph."""

    graph = StateGraph(FinancialAgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("validator", validator_node)
    graph.add_node("retry", retry_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "validator")

    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "retry": "retry",
            "finish": END,
        },
    )

    graph.add_edge("retry", "retrieval")

    return graph.compile()


# --------------------------------------------------
# 10. Public function for FastAPI
# --------------------------------------------------

def ask_agent(question: str) -> dict[str, Any]:
    """
    Process a financial question and return a structured result.

    Compatible with a FastAPI endpoint expecting:
    answer, sources, retrieved_docs, latency,
    validation_status, and agents_invoked.
    """

    global _agent

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if _agent is None:
        _agent = build_agent()

    logger.info("Starting financial agent workflow.")

    result = _agent.invoke(
        {
            "question": question.strip(),
            "retry_count": 0,
            "agent_trace": [],
        }
    )

    return {
        "question": result.get("question", question.strip()),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "retrieved_docs": result.get("retrieved_docs", []),
        "latency": result.get("latency", 0.0),
        "validation_status": result.get(
            "validation_status", "unknown"
        ),
        "agents_invoked": result.get("agent_trace", []),
    }