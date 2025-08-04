"""Ollama provider for LLM interactions.

This module provides a provider for interacting with local Ollama models.
"""

import json
import logging
from typing import Dict, Any, AsyncGenerator

import httpx

from src.llm.providers.base import BaseLLMProvider
from src.llm.providers import register_provider

logger = logging.getLogger(__name__)


@register_provider("ollama")
class OllamaProvider(BaseLLMProvider):
    """Provider for Ollama API.

    This class handles communication with the Ollama API for chat completions.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 60,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            base_url: Base URL for the Ollama API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        logger.debug(f"Initialized OllamaProvider with base_url: {base_url}")

    async def chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a chat completion.

        Args:
            payload: Chat completion parameters

        Returns:
            Chat completion response
        """
        # Extract parameters from payload
        model = payload.get("model", "phi3.5:3.8b")  # Default model
        messages = payload.get("messages", [])
        temperature = payload.get("temperature", 0.7)
        max_tokens = payload.get("max_tokens", 1024)
        stream = payload.get("stream", False)

        # Convert messages to Ollama format
        prompt = self._format_messages(messages)

        # Prepare Ollama API payload
        ollama_payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream,
            "options": {
                "num_ctx": 4096,  # Default context window
            },
        }

        # Handle streaming vs non-streaming
        if stream:
            return self._stream_response(ollama_payload)
        else:
            return await self._generate_response(ollama_payload)

    async def _generate_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a non-streaming response.

        Args:
            payload: Ollama API payload

        Returns:
            Formatted response
        """
        try:
            url = f"{self.base_url}/api/generate"
            response = await self.client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()

            # Convert Ollama response to OpenAI-like format for compatibility
            return {
                "id": "ollama-" + payload["model"],
                "object": "chat.completion",
                "created": 0,  # Ollama doesn't provide timestamp
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": data.get("response", ""),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0)
                    + data.get("eval_count", 0),
                },
            }

        except Exception as e:
            logger.error(f"Error in Ollama API call: {e}")
            raise

    async def _stream_response(
        self, payload: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Stream response from Ollama API.

        Args:
            payload: Ollama API payload

        Yields:
            Streamed response chunks
        """
        url = f"{self.base_url}/api/generate"

        try:
            async with self.client.stream(
                "POST", url, json=payload, timeout=None
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)

                        # Format in OpenAI-like streaming format
                        chunk = {
                            "choices": [
                                {
                                    "delta": {
                                        "content": data.get("response", ""),
                                    },
                                    "finish_reason": (
                                        "stop" if data.get("done", False) else None
                                    ),
                                    "index": 0,
                                }
                            ],
                            "model": payload["model"],
                        }

                        yield f"data: {json.dumps(chunk)}\n\n"

                        if data.get("done", False):
                            yield "data: [DONE]\n\n"

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming response: {line}")

        except Exception as e:
            logger.error(f"Error in Ollama streaming: {e}")
            raise

    def _format_messages(self, messages: list) -> str:
        """Format messages for Ollama API.

        Args:
            messages: List of message objects

        Returns:
            Formatted prompt string
        """
        formatted_prompt = ""

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                formatted_prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted_prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_prompt += f"<|assistant|>\n{content}\n"

        # Add final assistant prompt
        if not formatted_prompt.endswith("<|assistant|>\n"):
            formatted_prompt += "<|assistant|>\n"

        return formatted_prompt

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
