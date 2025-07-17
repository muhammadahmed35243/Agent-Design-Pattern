from dotenv import load_dotenv
import os
import requests  # You also forgot to import this

load_dotenv()  # Load variables from .env file 

API_KEY = os.getenv("GROK_API")  # Double-check if your .env variable is correct
BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def chat_with_grok(model: str, messages: list, temperature: float = 0.7):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }

    try:
        response = requests.post(BASE_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
