import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app

client = TestClient(app)

def test_successful_response_with_question():
    response = client.post("/ask", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]
    assert isinstance(data["choices"][0]["message"]["content"], str)

def test_successful_response_with_messages():
    messages = [{"role": "user", "content": "Tell me a joke."}]
    response = client.post("/ask", json={"messages": messages})
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]
    assert isinstance(data["choices"][0]["message"]["content"], str)

def test_invalid_request_missing_question_and_messages():
    response = client.post("/ask", json={"invalid": "data"})
    assert response.status_code == 200  
    data = response.json()
    assert "error" in data
