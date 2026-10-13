"""
Financial Market Intelligence
=============================

Streamlit frontend for the Agentic RAG application.

Backend:
FastAPI

LLM:
Groq

Retrieval:
FAISS + BM25

Orchestration:
LangGraph
"""

import os
import uuid

import requests
import streamlit as st


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Financial Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. API CONFIGURATION
# ==================================================

def get_api_base_url():
    """
    Resolve the backend URL from Streamlit secrets,
    environment variables, or the local development default.
    """

    try:
        configured_url = st.secrets.get("API_BASE_URL", "")
    except Exception:
        configured_url = ""

    configured_url = configured_url or os.getenv("API_BASE_URL")

    return (
        configured_url or "http://127.0.0.1:8000"
    ).rstrip("/")


API_BASE_URL = get_api_base_url()


# ==================================================
# 3. SESSION STATE
# ==================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_request_id" not in st.session_state:
    st.session_state.last_request_id = None

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


# ==================================================
# 4. API HELPER FUNCTIONS
# ==================================================

def check_api_health():
    """Check whether the FastAPI backend is available."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def ask_financial_question(question: str):
    """Send a financial question to the FastAPI backend."""

    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
    }

    response = requests.post(
        f"{API_BASE_URL}/query",
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


def submit_feedback(request_id, rating, comment):
    """Submit feedback for a response."""

    payload = {
        "request_id": request_id,
        "rating": rating,
        "comment": comment.strip() or None,
    }

    response = requests.post(
        f"{API_BASE_URL}/feedback",
        json=payload,
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


# ==================================================
# 5. AGENT WORKFLOW VISUALIZATION
# ==================================================

AGENT_DETAILS = {
    "planner": {
        "title": "Planner Agent",
        "icon": "🧠",
        "description": (
            "Receives the financial question, checks that it is not "
            "empty, and prepares it for retrieval."
        ),
    },
    "financial_rag": {
        "title": "Financial RAG Agent",
        "icon": "🔍",
        "description": (
            "Retrieves financial evidence using FAISS semantic search "
            "and BM25 keyword search, then generates a response using Groq."
        ),
    },
    "validator": {
        "title": "Validation Agent",
        "icon": "✅",
        "description": (
            "Checks whether an answer and retrieved evidence are available. "
            "The current validator does not verify every claim or financial "
            "figure against the source text."
        ),
    },
    "retry": {
        "title": "Retry Agent",
        "icon": "🔄",
        "description": (
            "Runs the configured retry path when the answer is empty "
            "and the retry limit has not been reached."
        ),
    },
}


def render_agent_workflow(response_data):
    """
    Display the actual executed LangGraph nodes.

    The current backend returns a completed trace rather than live
    intermediate events.
    """

    agents_invoked = response_data.get("agents_invoked", [])

    validation_status = response_data.get(
        "validation_status",
        "unknown",
    )

    sources = response_data.get("sources", [])

    retrieved_docs = response_data.get(
        "retrieved_docs",
        [],
    )

    with st.expander(
        "🤖 Agent Workflow — How this answer was generated",
        expanded=True,
    ):

        if not agents_invoked:
            st.info("No agent execution trace was returned.")
            return

        st.caption(
            "The steps below represent the LangGraph nodes "
            "reported by the backend."
        )

        # Display the actual node execution order.
        for step_number, agent_name in enumerate(
            agents_invoked,
            start=1,
        ):

            details = AGENT_DETAILS.get(
                agent_name,
                {
                    "title": agent_name.replace("_", " ").title(),
                    "icon": "⚙️",
                    "description": "Workflow node executed.",
                },
            )

            with st.container(border=True):

                st.markdown(
                    f"### {details['icon']} "
                    f"Step {step_number}: {details['title']}"
                )

                st.write(details["description"])

                if agent_name == "planner":

                    st.markdown("**Question received:**")

                    st.write(
                        response_data.get("question", "")
                    )

                    st.caption(
                        "Output: question prepared for retrieval."
                    )

                elif agent_name == "financial_rag":

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Evidence chunks",
                        len(retrieved_docs),
                    )

                    col2.metric(
                        "Source records",
                        len(sources),
                    )

                    st.markdown("**Retrieval process**")

                    st.write(
                        "1. FAISS performs semantic retrieval."
                    )

                    st.write(
                        "2. BM25 performs keyword retrieval."
                    )

                    st.write(
                        "3. The ensemble combines retrieval results."
                    )

                    st.write(
                        "4. Groq generates the final answer from the "
                        "retrieved context."
                    )

                    if sources:

                        with st.expander(
                            "View sources used by the RAG pipeline"
                        ):

                            for index, source in enumerate(
                                sources,
                                start=1,
                            ):

                                document = source.get(
                                    "document",
                                    "Unknown document",
                                )

                                entity = source.get(
                                    "entity",
                                    "Unknown entity",
                                )

                                year = source.get(
                                    "year",
                                    "Unknown year",
                                )

                                page = source.get(
                                    "page_no",
                                    "Unknown page",
                                )

                                st.markdown(
                                    f"**Source {index}: {document}**"
                                )

                                st.caption(
                                    f"Entity: {entity} | "
                                    f"Year: {year} | "
                                    f"Page: {page}"
                                )

                                preview = source.get(
                                    "content_preview",
                                    "",
                                )

                                if preview:
                                    st.text(preview)

                                st.divider()

                elif agent_name == "validator":

                    if validation_status == "evidence_available":

                        st.success(
                            "The answer and retrieved evidence are present."
                        )

                    elif validation_status == "insufficient_evidence":

                        st.warning(
                            "The answer was generated, but evidence "
                            "was marked insufficient."
                        )

                    elif validation_status == "failed":

                        st.error(
                            "The workflow marked answer generation as failed."
                        )

                    else:

                        st.info(
                            f"Validation status: {validation_status}"
                        )

                    st.caption(
                        "This is an availability check, not a full "
                        "semantic grounding evaluation."
                    )

                elif agent_name == "retry":

                    st.warning(
                        "The workflow entered its retry path."
                    )

        st.divider()

        st.markdown("### 📊 Workflow Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Executed nodes",
            len(agents_invoked),
        )

        col2.metric(
            "Retrieved chunks",
            len(retrieved_docs),
        )

        col3.metric(
            "Latency",
            f"{response_data.get('latency', 0):.2f} sec",
        )

        st.markdown("**Execution order**")

        st.code(
            " → ".join(agents_invoked),
            language=None,
        )


# ==================================================
# 6. SOURCE VIEWER
# ==================================================

def render_sources(response_data):
    """Display retrieved source metadata."""

    sources = response_data.get("sources", [])

    with st.expander("📚 View Financial Sources"):

        if not sources:
            st.info("No source metadata was returned.")
            return

        for index, source in enumerate(sources, start=1):

            document = source.get(
                "document",
                "Unknown document",
            )

            entity = source.get(
                "entity",
                "Unknown entity",
            )

            year = source.get(
                "year",
                "Unknown year",
            )

            page = source.get(
                "page_no",
                "Unknown page",
            )

            sector = source.get(
                "sector",
                "Unknown sector",
            )

            st.markdown(
                f"**{index}. {document}**"
            )

            st.caption(
                f"Entity: {entity} | "
                f"Year: {year} | "
                f"Sector: {sector} | "
                f"Page: {page}"
            )

            preview = source.get(
                "content_preview",
                "",
            )

            if preview:
                st.text(preview)

            st.divider()


# ==================================================
# 7. RETRIEVED EVIDENCE VIEWER
# ==================================================

def render_retrieved_evidence(response_data):
    """Display text snippets retrieved by the RAG pipeline."""

    documents = response_data.get("retrieved_docs", [])

    with st.expander("🔎 View Retrieved Evidence"):

        if not documents:
            st.info("No retrieved evidence was returned.")
            return

        for index, document in enumerate(
            documents,
            start=1,
        ):

            st.markdown(f"**Evidence Chunk {index}**")

            st.text(document)

            st.divider()


# ==================================================
# 8. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("📊 Financial Intelligence")

    st.caption(
        "Agentic RAG for financial question answering."
    )

    st.divider()

    st.subheader("Backend Status")

    if check_api_health():
        st.success("FastAPI connected")
    else:
        st.error("FastAPI unavailable")

    st.caption(f"Backend URL: {API_BASE_URL}")

    st.divider()

    st.subheader("Technology Stack")

    st.markdown(
        """
        - **LLM:** Groq
        - **Agents:** LangGraph
        - **Vector search:** FAISS
        - **Keyword search:** BM25
        - **Embeddings:** Hugging Face
        - **Backend:** FastAPI
        - **Frontend:** Streamlit
        """
    )

    st.divider()

    st.subheader("Example Questions")

    example_questions = [
        "What financial information is available for BMW in 2022?",
        "What information is available about Nvidia cash flows?",
        "What financial information is available for Mercedes-Benz in 2022?",
    ]

    for index, example in enumerate(
        example_questions,
    ):

        if st.button(
            example,
            key=f"example_{index}",
            use_container_width=True,
        ):

            st.session_state.pending_question = example

            st.rerun()

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.session_id = str(uuid.uuid4())

        st.session_state.last_request_id = None

        st.session_state.last_response = None

        st.session_state.feedback_submitted = False

        st.rerun()


# ==================================================
# 9. MAIN HEADER
# ==================================================

st.title("📈 Financial Market Intelligence")

st.markdown(
    """
    Ask questions about financial reports and company performance.

    The system retrieves financial evidence, generates an answer,
    and displays the agent workflow used to produce it.
    """
)

st.info(
    "Answers are limited by the financial documents available in the "
    "dataset. Verify important figures against the original reports."
)


# ==================================================
# 10. DISPLAY CHAT HISTORY
# ==================================================

for message_index, message in enumerate(
    st.session_state.messages,
):

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("response")
        ):

            response_data = message["response"]

            render_agent_workflow(response_data)

            render_sources(response_data)

            render_retrieved_evidence(response_data)

            with st.expander("⚙️ Request Details"):

                st.write(
                    "**Request ID:**",
                    response_data.get("request_id", "Unknown"),
                )

                st.write(
                    "**Session ID:**",
                    response_data.get("session_id", "Unknown"),
                )

                st.write(
                    "**Validation status:**",
                    response_data.get(
                        "validation_status",
                        "unknown",
                    ),
                )

                st.write(
                    "**Latency:**",
                    f"{response_data.get('latency', 0):.2f} seconds",
                )


# ==================================================
# 11. QUESTION INPUT
# ==================================================

pending_question = st.session_state.pop(
    "pending_question",
    None,
)

typed_question = st.chat_input(
    "Ask a financial question..."
)

question = pending_question or typed_question


# ==================================================
# 12. PROCESS QUESTION
# ==================================================

if question:

    question = question.strip()

    if not question:

        st.warning("Please enter a question.")

    elif len(question) > 2000:

        st.warning(
            "Please limit your question to 2000 characters."
        )

    else:

        # Add the user's question to conversation history.
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Agents are processing your financial question..."
            ):

                try:

                    result = ask_financial_question(question)

                    answer = result.get(
                        "answer",
                        "No answer was returned.",
                    )

                    st.markdown(answer)

                    # Save the complete API response so that the
                    # workflow can be rendered during subsequent reruns.
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "response": result,
                        }
                    )

                    st.session_state.last_request_id = result.get(
                        "request_id"
                    )

                    st.session_state.last_response = result

                    st.session_state.feedback_submitted = False

                    st.rerun()

                except requests.HTTPError as exc:

                    status_code = (
                        exc.response.status_code
                        if exc.response is not None
                        else "unknown"
                    )

                    detail = "The API rejected the request."

                    if exc.response is not None:

                        try:
                            detail = exc.response.json().get(
                                "detail",
                                detail,
                            )

                        except ValueError:
                            pass

                    st.error(
                        f"API error ({status_code}): {detail}"
                    )

                except requests.Timeout:

                    st.error(
                        "The request timed out. The backend may be "
                        "starting up or processing a slow request. "
                        "Please try again."
                    )

                except requests.RequestException:

                    st.error(
                        "Could not connect to FastAPI. Check the backend "
                        "URL and confirm that the API is running."
                    )

                except Exception:

                    st.error(
                        "An unexpected application error occurred. "
                        "Check the Streamlit and FastAPI logs."
                    )


# ==================================================
# 13. FEEDBACK
# ==================================================

if st.session_state.last_response:

    st.divider()

    st.subheader("⭐ Rate This Answer")

    if st.session_state.feedback_submitted:

        st.success(
            "Thank you! Your feedback has been submitted."
        )

    else:

        with st.form("feedback_form"):

            rating = st.slider(
                "How useful was this answer?",
                min_value=1,
                max_value=5,
                value=4,
            )

            comment = st.text_area(
                "Additional feedback (optional)",
                placeholder=(
                    "Was the answer accurate, relevant, "
                    "and supported by its sources?"
                ),
            )

            submitted = st.form_submit_button(
                "Submit Feedback",
                use_container_width=True,
            )

            if submitted:

                request_id = st.session_state.last_request_id

                if not request_id:

                    st.error(
                        "The request ID is missing. "
                        "Ask a new question before submitting feedback."
                    )

                else:

                    try:

                        feedback_result = submit_feedback(
                            request_id=request_id,
                            rating=rating,
                            comment=comment,
                        )

                        st.session_state.feedback_submitted = True

                        st.success(
                            feedback_result.get(
                                "message",
                                "Feedback submitted successfully.",
                            )
                        )

                        st.rerun()

                    except requests.HTTPError as exc:

                        st.error(
                            f"Feedback request failed: HTTP "
                            f"{exc.response.status_code if exc.response else 'unknown'}."
                        )

                    except requests.RequestException:

                        st.error(
                            "Could not submit feedback. "
                            "Please check the backend connection."
                        )


# ==================================================
# 14. FOOTER
# ==================================================

st.divider()

st.caption(
    "Financial Market Intelligence | "
    "FastAPI · Streamlit · LangGraph · Groq · FAISS · Hugging Face"
)