from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify that the API root endpoint responds successfully."""
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert "status" in data


def test_health_endpoint():
    """Verify that the health endpoint responds successfully."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"


@patch("main.ask_agent")
def test_query_success(mock_ask_agent):
    """Verify that a valid financial question returns an answer."""
    mock_ask_agent.return_value = {
        "question": "What financial information is available for BMW in 2022?",
        "answer": "The retrieved documents contain BMW financial information for 2022.",
        "sources": [],
        "retrieved_docs": ["BMW annual report excerpt"],
        "latency": 0.25,
        "validation_status": "evidence_available",
        "agents_invoked": ["planner", "financial_rag", "validator"],
    }

    response = client.post(
        "/query",
        json={
            "question": "What financial information is available for BMW in 2022?"
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert "BMW" in data["answer"]
    assert data["validation_status"] == "evidence_available"
    assert "planner" in data["agents_invoked"]

    mock_ask_agent.assert_called_once()


def test_query_rejects_empty_question():
    """Verify that an empty question is rejected."""
    response = client.post(
        "/query",
        json={"question": ""},
    )

    assert response.status_code == 422


@patch("main.ask_agent")
def test_query_handles_agent_error(mock_ask_agent):
    """Verify that an agent exception is handled by the API."""
    mock_ask_agent.side_effect = RuntimeError("Simulated agent failure")

    response = client.post(
        "/query",
        json={"question": "Analyze BMW financial performance"},
    )

    assert response.status_code == 503