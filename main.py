"""
Financial Market Intelligence
=============================

FastAPI backend for the Agentic RAG application.

Architecture:
Streamlit -> FastAPI -> LangGraph -> Hybrid RAG -> Groq

Features:
- Financial question answering
- Agent execution trace
- Source metadata
- Health check
- User feedback
- Request and session IDs
"""

import logging
import time
import uuid

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import ask_agent


# ==================================================
# 1. CONFIGURATION
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Financial Market Intelligence API",
    description=(
        "An Agentic RAG application for financial question answering "
        "using LangGraph, Groq, FAISS, BM25, and Hugging Face embeddings."
    ),
    version="1.0.0",
)


# ==================================================
# 2. CORS CONFIGURATION
# ==================================================

# Suitable for local development and a public Streamlit frontend.
# Restrict allow_origins to your actual frontend domain for production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ==================================================
# 3. PYDANTIC MODELS
# ==================================================

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Financial question.",
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Optional conversation session identifier.",
    )


class SourceItem(BaseModel):
    source: Optional[str] = None
    row: Optional[int] = None
    document: Optional[str] = None
    page_no: Optional[str] = None
    year: Optional[str] = None
    sector: Optional[str] = None
    entity: Optional[str] = None
    content_preview: Optional[str] = None


class QueryResponse(BaseModel):
    request_id: str
    session_id: str
    question: str
    answer: str
    sources: list[SourceItem]
    retrieved_docs: list[str]
    agents_invoked: list[str]
    validation_status: str
    latency: float


class FeedbackRequest(BaseModel):
    request_id: str

    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating between 1 and 5.",
    )

    comment: Optional[str] = Field(
        default=None,
        max_length=2000,
    )


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str


# ==================================================
# 4. TEMPORARY FEEDBACK STORAGE
# ==================================================

# Feedback is stored in memory for this demonstration.
# It will be lost when the API process restarts.
# Use a database for persistent storage.

feedback_records: list[dict] = []


# ==================================================
# 5. ROOT ENDPOINT
# ==================================================

@app.get("/")
def home():
    return {
        "application": "Financial Market Intelligence",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "feedback": "/feedback",
            "documentation": "/docs",
        },
    }


# ==================================================
# 6. HEALTH CHECK
# ==================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Financial Market Intelligence API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ==================================================
# 7. FINANCIAL QUESTION ENDPOINT
# ==================================================

@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_financial_data(request: QueryRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    request_id = str(uuid.uuid4())

    session_id = request.session_id or str(uuid.uuid4())

    start_time = time.perf_counter()

    logger.info(
        "Request %s started. Session: %s",
        request_id,
        session_id,
    )

    try:
        # Run the existing LangGraph workflow.
        result = ask_agent(question)

        answer = result.get("answer", "")

        if not isinstance(answer, str) or not answer.strip():
            raise HTTPException(
                status_code=502,
                detail="The financial agent returned an empty answer.",
            )

        sources = result.get("sources", [])

        retrieved_docs = result.get("retrieved_docs", [])

        agents_invoked = result.get("agents_invoked", [])

        latency = result.get(
            "latency",
            time.perf_counter() - start_time,
        )

        response = QueryResponse(
            request_id=request_id,
            session_id=session_id,
            question=question,
            answer=answer,
            sources=sources,
            retrieved_docs=retrieved_docs,
            agents_invoked=agents_invoked,
            validation_status=result.get(
                "validation_status",
                "unknown",
            ),
            latency=round(float(latency), 4),
        )

        logger.info(
            "Request %s completed in %.2f seconds.",
            request_id,
            latency,
        )

        return response

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Request %s failed.",
            request_id,
        )

        # Avoid exposing API keys or internal exception details.
        raise HTTPException(
            status_code=503,
            detail=(
                "The financial intelligence service could not "
                "process your question. Please try again."
            ),
        ) from exc


# ==================================================
# 8. FEEDBACK ENDPOINT
# ==================================================

@app.post(
    "/feedback",
    response_model=FeedbackResponse,
)
def submit_feedback(request: FeedbackRequest):

    feedback_id = str(uuid.uuid4())

    record = {
        "feedback_id": feedback_id,
        "request_id": request.request_id,
        "rating": request.rating,
        "comment": request.comment,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    feedback_records.append(record)

    logger.info(
        "Feedback received for request %s.",
        request.request_id,
    )

    return FeedbackResponse(
        feedback_id=feedback_id,
        message="Feedback submitted successfully.",
    )


# ==================================================
# 9. RUN SERVER
# ==================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )