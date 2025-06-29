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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it to a valid OpenRouter API key.")

data_dir = os.path.join(os.path.dirname(__file__), "data")
documents = SimpleDirectoryReader(data_dir).load_data()

shipping_content = ""
for doc in documents:
    shipping_content += doc.text.lower()

Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

Settings.llm = None

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

index = VectorStoreIndex.from_documents(documents)

def query_openrouter(user_input: str):
    try:
        query_engine = index.as_query_engine(similarity_top_k=3)
        retrieved_content = query_engine.query(user_input)
        
        location_match = re.search(r'\b(?:to|for|in)\s+([A-Za-z\s]+?)(?:\s|$)', user_input, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).lower().strip()
            invalid_locations = {"moon", "south pole"}
            if location in invalid_locations:
                return f"We're sorry, we don't have shipping details for {location_match.group(1)} right now. Please reach out to us directly at support@artisanproducts.com or call +1-800-555-1234 for assistance."
            if location not in shipping_content:
                return f"Thanks for your question! We couldn't find specific information about {location_match.group(1)}. For the most accurate and up-to-date details, please contact us directly."
        
        prompt = (
            f"You are the Artisan Products team, specializing in handcrafted items. Respond in a collective, authoritative tone using 'We' to reflect the team.\n\n"
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
                {"role": "system", "content": "You are the Artisan Products team, specializing in handcrafted items. Respond in a collective, authoritative tone using 'We' to reflect the team."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        import requests
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 401:
            print(f"Query failed: Authentication error - {e}")
            return "We're sorry, we're having trouble accessing the information right now. Please contact us directly at support@artisanproducts.com for assistance."
        print(f"Query failed: {e}")
        return "Sorry, we couldn't process your question right now."

query_engine = type("QueryEngineWrapper", (), {"query": staticmethod(query_openrouter)})