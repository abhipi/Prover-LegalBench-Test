"""Sample script to call deepseek/deepseek-r1 model via OpenRouter."""

import asyncio
import os
from dotenv import load_dotenv
import httpx

load_dotenv(override=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


async def call_deepseek_r1(prompt: str, api_key: str = None) -> dict:
    """
    Call deepseek/deepseek-r1 model via OpenRouter.
    
    Args:
        prompt: The user's message/prompt
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
    
    Returns:
        Response dictionary from OpenRouter API
    """
    if not api_key:
        api_key = OPENROUTER_API_KEY
    
    if not api_key:
        raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com",  # Optional: for tracking
        "X-Title": "Voice Duolingo",  # Optional: for tracking
    }
    
    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()


async def main():
    """Example usage."""
    prompt = "Hello! Can you explain what you are?"
    
    try:
        result = await call_deepseek_r1(prompt)
        
        # Extract the response text
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            content = message.get("content", "")
            print("Response:")
            print(content)
            print("\n---")
            print("Full response:", result)
        else:
            print("Unexpected response format:", result)
            
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
