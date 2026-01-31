import os
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'openai'. Install with: pip install openai"
    ) from e


class OpenRouterChatClient:
    """Small wrapper for OpenRouter's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "deepseek/deepseek-prover-v2",
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Export it or pass api_key=..."
            )

        default_headers: Dict[str, str] = {}
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if x_title:
            default_headers["X-Title"] = x_title

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            default_headers=default_headers if default_headers else None,
        )
        self.model = model

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
