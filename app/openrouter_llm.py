import requests
from typing import List, Optional
from llama_index.core.llms.llm import LLM 



class OpenRouterLLM(LLM):
    def __init__(self, api_key: str, model: str = "openai/gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def metadata(self) -> dict:
        return {"name": "OpenRouterLLM", "model": self.model}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": messages,
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}
