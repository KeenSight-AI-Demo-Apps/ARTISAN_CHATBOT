# 🧵 Artisan Support Chatbot

A simple RAG-based chatbot for artisan marketplaces using LlamaIndex, FastAPI, Streamlit, and OpenRouter.

## 🔧 Features

- Loads support info from 'app/data/'
- Uses 'HuggingFaceEmbedding' for text vectorization
- Retrieves relevant context via LlamaIndex
- Sends prompt + context to OpenRouter ('gpt-4o')
- Streamlit frontend to interact with chatbot
- FastAPI backend with '/ask' endpoint

## 🚀 Setup


# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key in a .env file
echo OPENROUTER_API_KEY=your_key_here > .env
