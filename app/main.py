from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from chatbot import query_engine
from typing import List, Dict
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Artisan AI Chatbot running"}

@app.get("/ask/models")
@app.get("/ask/chat/completions/models")
async def get_models():
    """Return a list of available models for OpenWebUI compatibility."""
    return {
        "data": [
            {
                "id": "openai/gpt-4o",
                "name": "GPT-4o",
                "object": "model",
                "created": 1699478378,
                "owned_by": "openrouter"
            }
        ],
        "object": "list"
    }

@app.post("/ask")
@app.post("/ask/chat/completions")
@app.post("/ask/chat/completions/chat/completions")
async def ask(request: Request):
    data = await request.json()
    # Handle both single question and OpenAI-compatible messages format
    if "question" in data:
        messages = [{"role": "user", "content": data["question"]}]
    elif "messages" in data:
        messages = data["messages"]
    else:
        return {"error": "Invalid request format. Provide 'question' or 'messages'."}
    
    response = query_engine.query(messages)
    # Return OpenAI-compatible response for OpenWebUI
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }
        ],
        "model": "openai/gpt-4o"
    }