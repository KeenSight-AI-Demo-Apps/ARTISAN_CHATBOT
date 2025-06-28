from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.chatbot import query_engine

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

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    user_input = data.get("question", "")
    response = query_engine.query(user_input)
    return {"answer": str(response)}
