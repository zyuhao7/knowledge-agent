from __future__ import annotations

from typing import Any

import httpx

from .config import Settings


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Small OpenAI-compatible chat client; works with OpenAI, DeepSeek, vLLM, Ollama's bridge, etc."""

    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise ValueError("OPENAI_API_KEY is required to use the LLM path")
        self.settings = settings

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = httpx.post(
                f"{self.settings.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            body = error.response.text[:500]
            raise LLMError(
                f"chat request failed: HTTP {error.response.status_code}: {body}"
            ) from error
        except httpx.HTTPError as error:
            raise LLMError(f"chat request failed: {error}") from error

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"chat response has no choices: {data}")
        return choices[0]["message"]
