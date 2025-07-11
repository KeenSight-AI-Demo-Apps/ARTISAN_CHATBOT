import os
import re
import json
import requests
from dotenv import load_dotenv
from typing import Dict, List, TypedDict
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI
from langgraph.graph import StateGraph, END


load_dotenv(override=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-proj-"):
    raise ValueError("A valid project-specific OpenAI API key (sk-proj-...) is required.")


client = OpenAI(api_key=OPENAI_API_KEY)


data_dir = os.path.join(os.path.dirname(__file__), "data")
documents = SimpleDirectoryReader(data_dir).load_data()


Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = None  


index = VectorStoreIndex.from_documents(documents)


class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    retrieved_content: str
    response: str


def retrieve_documents(state: ChatState) -> ChatState:
    user_input = state["messages"][-1]["content"]
    query_engine = index.as_query_engine(similarity_top_k=3)
    retrieved_content = str(query_engine.query(user_input))
    return {"retrieved_content": retrieved_content}


def check_location(state: ChatState) -> ChatState:
    user_input = state["messages"][-1]["content"]
    retrieved_content = state["retrieved_content"]

    location_match = re.search(r'\b(?:to|for|in)\s+(\w+)\b', user_input, re.IGNORECASE)
    if location_match:
        location = location_match.group(1).lower()
        valid_locations = {"united states", "canada", "australia", "europe", "asia"}
        if not retrieved_content or (location not in retrieved_content.lower() and location not in valid_locations):
            return {
                "response": (
                    "I'm sorry, I don't have the information to answer that right now. "
                    "Please contact support@example.com or call 1-800-ARTISAN for assistance."
                )
            }
    return state


def generate_response(state: ChatState) -> ChatState:
    if state.get("response"):  
        return state

    retrieved_content = state["retrieved_content"]
    user_input = state["messages"][-1]["content"]

    prompt = (
        "You are an expert support agent for artisan products. "
        "Answer the following question based on the provided context from the knowledge base.\n\n"
        f"Context:\n{retrieved_content}\n\n"
        f"Question: {user_input}\n\n"
        "Provide a concise and accurate response."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {"role": "system", "content": "You are an expert support agent for artisan products."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return {"response": response.choices[0].message.content}

    except Exception as e:
        print("❌ Error calling OpenAI:", str(e))
        return {"response": "Sorry, I couldn't process your question right now."}


workflow = StateGraph(ChatState)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("check_location", check_location)
workflow.add_node("generate_response", generate_response)

workflow.add_edge("retrieve_documents", "check_location")
workflow.add_edge("check_location", "generate_response")
workflow.add_edge("generate_response", END)

workflow.set_entry_point("retrieve_documents")
graph = workflow.compile()


def query_openai(messages: List[Dict[str, str]]) -> str:
    state = {"messages": messages, "retrieved_content": "", "response": ""}
    result = graph.invoke(state)
    return result["response"]


query_engine = type("QueryEngineWrapper", (), {"query": staticmethod(query_openai)})
