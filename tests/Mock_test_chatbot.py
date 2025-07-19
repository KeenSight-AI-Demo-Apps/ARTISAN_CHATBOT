import pytest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.chatbot import retrieve_documents, check_location, generate_response, query_openai


@pytest.fixture
def sample_state():
    return {
        "messages": [{"role": "user", "content": "Tell me about Artisan AI in Canada"}],
        "retrieved_content": "Artisan AI is available in Canada and US",
        "response": ""
    }


def test_check_location_valid(sample_state):
    result = check_location(sample_state)
    assert isinstance(result, dict)
    assert "response" not in result or result.get("response") == ""


def test_check_location_invalid():
    invalid_state = {
        "messages": [{"role": "user", "content": "Tell me about Artisan AI in Nowhereville"}],
        "retrieved_content": "",
        "response": ""
    }
    result = check_location(invalid_state)
    assert "response" in result
    assert "I'm sorry" in result["response"]


@patch("app.chatbot.index")
def test_retrieve_documents(mock_index, sample_state):
    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = "Mocked document content."
    mock_index.as_query_engine.return_value = mock_query_engine

    result = retrieve_documents(sample_state)
    assert "retrieved_content" in result
    assert "Mocked document content." in result["retrieved_content"]


@patch("app.chatbot.client.chat.completions.create")
def test_generate_response(mock_chat_create):
    mock_chat_create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Mocked response from OpenAI"))]
    )
    state = {
        "messages": [{"role": "user", "content": "What is Artisan AI?"}],
        "retrieved_content": "Some context content",
        "response": ""
    }
    result = generate_response(state)
    assert "response" in result
    assert result["response"] == "Mocked response from OpenAI"


@patch("app.chatbot.graph.invoke")
def test_query_openai(mock_invoke):
    mock_invoke.return_value = {"response": "Mocked OpenAI response."}
    messages = [{"role": "user", "content": "Tell me a joke."}]
    response = query_openai(messages)
    assert response == "Mocked OpenAI response."
