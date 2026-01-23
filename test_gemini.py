
import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")

print(f"Testing Gemini API with Model: {MODEL}")
print(f"API Key present: {bool(API_KEY)}")

async def test_gemini():
    # List models first
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"Listing models from {list_url}...")
            response = await client.get(list_url)
            
            if response.status_code == 200:
                data = response.json()
                print("Available Models:")
                for m in data.get("models", []):
                    if "generateContent" in m.get("supportedGenerationMethods", []):
                        print(f" - {m['name']}")
            else:
                print(f"Failed to list models: {response.text}")
                
            # Then try to generate content with current configured model
            print(f"\nTesting generation with configured model: {MODEL}")
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
            
            request_body = {
                "contents": [{
                    "parts": [{"text": "Hello"}]
                }]
            }
            
            print(f"Sending request to {api_url}...")
            response = await client.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=request_body
            )
            print(f"Response Status: {response.status_code}")
            if response.status_code != 200:
                print("Error:", response.text)
                
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini())
