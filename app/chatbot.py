import os
import re
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI  

data_dir = os.path.join(os.path.dirname(__file__), "data")
documents = SimpleDirectoryReader(data_dir).load_data()


Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")


Settings.llm = None


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")


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
            
            if not retrieved_content or location not in str(retrieved_content).lower():
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
        print(f"Query failed: {e}")
        return "Sorry, I couldn't process your question right now."


query_engine = type("QueryEngineWrapper", (), {"query": staticmethod(query_openrouter)})