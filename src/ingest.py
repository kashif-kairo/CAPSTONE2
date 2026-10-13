"""
Financial Market Intelligence
-----------------------------
Loads financial QA data from CSV, converts financial evidence into
LangChain Documents, splits documents into chunks, creates Hugging Face
embeddings, and saves the FAISS vector index.

Stack:
- Pandas
- Hugging Face sentence-transformers
- LangChain
- FAISS
"""

import logging
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
INDEX_DIR = BASE_DIR / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Load and validate financial dataset
# --------------------------------------------------

def load_financial_documents(csv_path: Path) -> list[Document]:
    """
    Load financial data from a CSV file and convert each row into
    a LangChain Document.

    The function identifies available columns rather than assuming
    that every dataset has the same schema.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            f"Please place the financial CSV file in: {DATA_DIR}"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"The dataset is empty: {csv_path}")

    # Normalize column names for easier matching.
    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    logger.info("Loaded dataset: %s", csv_path.name)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Available columns: %s", list(df.columns))

    # Candidate columns that may contain actual financial evidence.
    evidence_candidates = [
        "context",
        "contexts",
        "document_text",
        "content",
        "passage",
        "text",
        "document",
    ]

    question_candidates = [
        "question",
        "instruction",
        "query",
    ]

    answer_candidates = [
        "answer",
        "answers",
        "output",
        "response",
    ]

    evidence_column = next(
        (col for col in evidence_candidates if col in df.columns),
        None,
    )

    question_column = next(
        (col for col in question_candidates if col in df.columns),
        None,
    )

    answer_column = next(
        (col for col in answer_candidates if col in df.columns),
        None,
    )

    # Do not silently index the answers as evidence when a separate
    # context/document field is available.
    if evidence_column is None and question_column is None:
        raise ValueError(
            "Could not identify a usable evidence or question column. "
            f"Available columns: {list(df.columns)}"
        )

    documents = []

    for row_number, row in df.iterrows():
        evidence_parts = []

        # Prefer the dataset's supporting financial context.
        if evidence_column:
            value = row.get(evidence_column)

            if pd.notna(value) and str(value).strip():
                evidence_parts.append(str(value).strip())

        # Some QA datasets have no context column. In that case,
        # use the question as a limited fallback, but do not add
        # its reference answer to the retrieval index.
        if not evidence_parts and question_column:
            value = row.get(question_column)

            if pd.notna(value) and str(value).strip():
                evidence_parts.append(
                    f"Financial question: {str(value).strip()}"
                )

        if not evidence_parts:
            continue

        # Preserve useful metadata without placing the reference
        # answer into the indexed evidence.
        metadata = {
            "source": csv_path.name,
            "row": int(row_number),
        }

        for column in [
            "document",
            "page",
            "page_no",
            "year",
            "sector",
            "entity",
        ]:
            if column in df.columns:
                value = row.get(column)

                if pd.notna(value):
                    metadata[column] = str(value)

        # Keep the question as metadata where available.
        if question_column:
            question_value = row.get(question_column)

            if pd.notna(question_value):
                metadata["question"] = str(question_value).strip()

        documents.append(
            Document(
                page_content="\n".join(evidence_parts),
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError(
            f"No usable financial documents were found in {csv_path.name}."
        )

    logger.info("Created %d financial documents.", len(documents))

    return documents


# --------------------------------------------------
# 3. Split documents into chunks
# --------------------------------------------------

def split_documents(documents: list[Document]) -> list[Document]:
    """Split financial evidence into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced no chunks.")

    logger.info("Created %d document chunks.", len(chunks))

    return chunks


# --------------------------------------------------
# 4. Create Hugging Face embeddings
# --------------------------------------------------

def create_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the local sentence-transformers embedding model."""

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# --------------------------------------------------
# 5. Build and save FAISS index
# --------------------------------------------------

def build_vectorstore(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Create a FAISS vectorstore from financial document chunks."""

    logger.info("Creating FAISS vector index...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    logger.info("FAISS index saved to: %s", INDEX_DIR)

    return vectorstore


# --------------------------------------------------
# 6. Main ingestion pipeline
# --------------------------------------------------

def ingest_financial_data() -> FAISS:
    """
    Complete ingestion pipeline:
    CSV -> Documents -> Chunks -> Embeddings -> FAISS
    """

    documents = load_financial_documents(TRAIN_FILE)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)

    logger.info("Financial data ingestion completed successfully.")

    return vectorstore


if __name__ == "__main__":
    ingest_financial_data()