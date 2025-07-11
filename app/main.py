from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from chatbot import query_engine
from typing import List, Dict

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
    return {"message": "Artisan AI Chatbot is running."}

@app.get("/ask/models")
@app.get("/ask/chat/completions/models")
def get_models():
    return {
        "data": [
            {
                "id": "openai/gpt-4o-2024-11-20",
                "name": "EyobBot",
                "object": "model",
                "created": 1699478378,
                "owned_by": "openrouter"
            }
        ],
        "object": "list"
    }

@app.post("/ask")
@app.post("/ask/chat/completions")
async def ask(request: Request):  
    data = await request.json()
    messages = [{"role": "user", "content": data.get("question")}] if "question" in data else data.get("messages")
    if not messages:
        return {"error": "Invalid request. Provide 'question' or 'messages'."}
    
    response = query_engine.query(messages)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }
        ],
        "model": "openai/gpt-4o-2024-11-20"
    }

