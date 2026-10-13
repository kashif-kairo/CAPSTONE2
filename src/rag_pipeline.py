"""
Financial Market Intelligence
-----------------------------
Retrieval-Augmented Generation pipeline.

LLM: Groq
Embeddings: Hugging Face
Vector database: FAISS
Retrieval: Hybrid search (FAISS + BM25)
Orchestration: Compatible with LangGraph agent
"""


import logging
import os
import time
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "faiss_index"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=False)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_K = 5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Initialize embeddings
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Load the same Hugging Face embedding model used during ingestion.
    """

    logger.info("Loading Hugging Face embedding model...")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# --------------------------------------------------
# 3. Initialize hybrid retriever
# --------------------------------------------------

def create_hybrid_retriever(vectorstore):
    """
    Combine semantic retrieval through FAISS with keyword retrieval
    through BM25.
    """

    logger.info("Creating hybrid retriever...")

    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": 15,
            "lambda_mult": 0.7,
        },
    )

    # Use documents already stored in FAISS for BM25.
    documents = list(vectorstore.docstore._dict.values())

    if not documents:
        raise ValueError(
            "The FAISS index contains no documents. "
            "Run python -m src.ingest first."
        )

    keyword_retriever = BM25Retriever.from_documents(
        documents=documents,
        k=RETRIEVAL_K,
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            semantic_retriever,
            keyword_retriever,
        ],
        weights=[0.6, 0.4],
    )

    logger.info("Hybrid retriever initialized successfully.")

    return hybrid_retriever


# --------------------------------------------------
# 4. Load RAG pipeline
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_pipeline():
    """
    Load FAISS, create hybrid retrieval, and initialize Groq.
    Cached so the model and index are not repeatedly initialized.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to the project .env file."
        )

    if not GROQ_MODEL:
        raise ValueError(
            "GROQ_MODEL is missing. Add a supported model ID to .env."
        )

    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_DIR}. "
            "Run python -m src.ingest first."
        )

    logger.info("Loading FAISS index...")

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        folder_path=str(INDEX_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = create_hybrid_retriever(vectorstore)

    logger.info("Initializing Groq LLM...")

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=1024,
        max_retries=2,
        timeout=60,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Financial Market Intelligence assistant.

Answer financial questions using the supplied retrieved evidence.

Rules:
1. Ground factual claims in the supplied context.
2. Never invent financial figures, dates, company results, or sources.
3. If the context does not contain sufficient evidence, clearly say:
   "The answer is not available in the provided financial context."
4. Distinguish facts from calculations or reasonable interpretations.
5. If the user asks for a calculation, explain the calculation when possible.
6. Use conversation history only to understand follow-up questions.
7. Do not present unsupported claims as financial facts.
8. Do not provide personalized investment recommendations.
9. Keep answers clear, professional, and relevant.
10. Mention available source information when it supports the answer.

Retrieved financial context:
{context}
""",
            ),
            (
                "human",
                """
Conversation history:
{chat_history}

Question:
{question}

Provide a grounded answer.
""",
            ),
        ]
    )

    # Return the components required by the RAG pipeline.
    logger.info("Financial RAG pipeline loaded successfully.")

    return {
        "llm": llm,
        "retriever": retriever,
        "prompt": prompt,
    }


# --------------------------------------------------
# 5. Ask a financial question
# --------------------------------------------------

def ask_question(
    chain,
    question: str,
    chat_history: str = "",
) -> dict:
    """
    Retrieve evidence and generate a grounded financial answer.

    The function name and response fields are kept compatible with
    the existing LangGraph agent.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    start_time = time.perf_counter()

    question = question.strip()

    # Support the pipeline dictionary returned by load_pipeline().
    if not isinstance(chain, dict):
        raise TypeError(
            "Expected the RAG pipeline dictionary returned by load_pipeline()."
        )

    retriever = chain["retriever"]
    llm = chain["llm"]
    prompt = chain["prompt"]

    # Retrieve relevant financial evidence.
    docs = retriever.invoke(question)

    context = "\n\n".join(
        f"Source {index}:\n{doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )

    # Generate the answer using retrieved context.
    messages = prompt.format_messages(
        context=context or "No relevant financial evidence was retrieved.",
        chat_history=chat_history or "No previous conversation.",
        question=question,
    )

    response = llm.invoke(messages)

    answer = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )

    latency = time.perf_counter() - start_time

    retrieved_docs = [
        doc.page_content[:500]
        for doc in docs
    ]

    sources = [
        {
            **doc.metadata,
            "content_preview": doc.page_content[:200],
        }
        for doc in docs
    ]

    logger.info(
        "Financial question answered in %.2f seconds.",
        latency,
    )

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "sources": sources,
        "latency": round(latency, 4),
    }