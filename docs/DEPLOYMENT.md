# 🚀 Deployment Steps for Artisan Chatbot


## Option A: Local Deployment (FastAPI + Uvicorn)

```bash
# Clone the repository
git clone https://github.com/your-username/artisan-chatbot
cd artisan-chatbot

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Use .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Create and edit the .env file
touch .env  # Or create manually on Windows
# Add the following inside .env:
# OPENAI_API_KEY=your_api_key_here

# Run the app
uvicorn main:app --reload

## Option B: Deployment to Render
# Push your code to GitHub

# On Render:
# 1. Create a new Web Service
# 2. Set build command:
pip install -r requirements.txt

# 3. Set start command:
uvicorn main:app --host=0.0.0.0 --port=10000

# 4. Add Environment Variables:
# OPENAI_API_KEY=your_api_key_here

# Done!
## Option C: Deployment with Docker
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

##Build & Run
# Build the Docker image
docker build -t artisan-chatbot .

# Run the container
docker run -p 8000:8000 --env-file .env artisan-chatbot
