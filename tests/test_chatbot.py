import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.chatbot import (
    retrieve_documents,
    check_location,
    generate_response,
    query_openai,
    ChatState,
)

@pytest.fixture
def sample_state():
    return {
        "messages": [{"role": "user", "content": "What is Artisan AI?"}],
        "retrieved_content": "",
        "response": ""
    }

def test_retrieve_documents(sample_state):
   
    result = retrieve_documents(sample_state)
    assert isinstance(result, dict)
    assert "retrieved_content" in result
    assert isinstance(result["retrieved_content"], str)
    assert len(result["retrieved_content"]) > 0  

def test_check_location_valid(sample_state):
    sample_state["messages"][-1]["content"] = "Tell me about Artisan AI in Canada"
    sample_state["retrieved_content"] = "Artisan AI is available in Canada and US"
    result = check_location(sample_state)
    
    assert isinstance(result, dict)
    if "response" in result:
       
        assert result["response"] == "" or not result["response"]

def test_check_location_invalid(sample_state):
    sample_state["messages"][-1]["content"] = "Tell me about Artisan AI in Atlantis"
    sample_state["retrieved_content"] = "Artisan AI is available only in US and Canada"
    result = check_location(sample_state)
    assert isinstance(result, dict)
   
    assert "response" in result
    assert isinstance(result["response"], str)
    assert "sorry" in result["response"].lower()

def test_generate_response(sample_state):
    
    ret = retrieve_documents(sample_state)
    sample_state.update(ret)
   
    result = generate_response(sample_state)
    assert isinstance(result, dict)
    assert "response" in result
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0

def test_query_openai():
    messages = [{"role": "user", "content": "Explain what a chatbot is in one sentence."}]
    response = query_openai(messages)
    assert isinstance(response, str)
    assert len(response) > 0
