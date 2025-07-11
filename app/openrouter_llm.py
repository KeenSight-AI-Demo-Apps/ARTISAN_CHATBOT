import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx"

if not OPENAI_API_KEY.startswith("sk-proj-"):
    raise ValueError("This script is meant for OpenAI project keys (sk-proj-...).")


client = OpenAI(api_key=OPENAI_API_KEY)

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Can you confirm this OpenAI key works?"}
        ],
        max_tokens=100
    )

    print("✅ Success! The key works. Response:")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ Failed to use the API key.")
    print(e)
