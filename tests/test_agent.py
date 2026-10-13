from unittest.mock import patch

from src.agent import ask_agent


@patch("src.agent.get_rag_chain")
@patch("src.agent.ask_question")
def test_agent_returns_answer(mock_ask_question, mock_get_rag_chain):
    """Verify that the agent returns a financial answer."""
    mock_get_rag_chain.return_value = object()

    mock_ask_question.return_value = {
        "answer": "BMW's retrieved annual report contains financial information.",
        "sources": [{"entity": "BMW", "year": 2022}],
        "retrieved_docs": ["BMW annual report excerpt"],
        "latency": 0.2,
    }

    result = ask_agent("What financial information is available for BMW?")

    assert result["answer"]
    assert result["question"]
    assert "planner" in result["agents_invoked"]
    assert "financial_rag" in result["agents_invoked"]
    assert "validator" in result["agents_invoked"]
    assert result["validation_status"] == "evidence_available"


import pytest


def test_agent_rejects_empty_question():
    """Verify that empty questions raise a ValueError."""
    with pytest.raises(ValueError, match="Question cannot be empty"):
        ask_agent("   ")

    