import os
import re
from dotenv import load_dotenv,find_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI


load_dotenv(override=True)
env_path = find_dotenv()
load_dotenv(dotenv_path=env_path)


api_key = os.getenv("OPENROUTER_API_KEY")


print("Loaded API key:", api_key)
print(" .env file loaded from:", env_path if env_path else " Not found")

data_dir = os.path.join(os.path.dirname(__file__), "data")
documents = SimpleDirectoryReader(data_dir).load_data()

Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

Settings.llm = None

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it to a valid OpenRouter API key.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

index = VectorStoreIndex.from_documents(documents)

def query_openrouter(user_input: str):
    try:
        
        query_engine = index.as_query_engine(similarity_top_k=3)
        retrieved_content = query_engine.query(user_input)
        
  
        location_match = re.search(r'\b(?:to|for|in)\s+(\w+)\b', user_input, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).lower()
           
            valid_locations = {"united states", "canada", "australia", "europe", "asia"}
            
            if not retrieved_content or (location not in str(retrieved_content).lower() and location not in valid_locations):
                return "I'm sorry, I don't have the information to answer that right now. In the meantime, you can reach our support team directly at support@example.com or call 1-80 for further assistance."
        
      
        prompt = (
            f"You are an expert support agent for artisan products. Answer the following question based on the provided context from the knowledge base:\n\n"
            f"Context: {retrieved_content}\n\n"
            f"Question: {user_input}\n\n"
            f"Provide a concise and accurate response based on the context."
        )

       
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Artisan Chatbot",
            },
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert support agent for artisan products."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        import requests
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 401:
            print(f"Query failed: Authentication error - {e}")
            return "I'm sorry, I couldn't process your question due to an authentication issue. Please contact support@example.com for assistance."
        print(f"Query failed: {e}")
        return "Sorry, I couldn't process your question right now."


query_engine = type("QueryEngineWrapper", (), {"query": staticmethod(query_openrouter)})