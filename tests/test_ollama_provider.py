"""Tests for the Ollama provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from src.llm.providers.ollama import OllamaProvider


@pytest.fixture
def mock_response():
    """Create a mock response for testing."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "model": "phi3.5:3.8b",
        "created_at": "2025-07-19T12:34:56Z",
        "response": "This is a test response from Ollama.",
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 123456789,
        "load_duration": 12345678,
        "prompt_eval_count": 10,
        "eval_count": 20,
        "eval_duration": 98765432,
    }
    return response


@pytest.mark.asyncio
async def test_chat_completion(mock_response):
    """Test generating a chat completion."""
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        provider = OllamaProvider(base_url="http://test.local:11434")
        payload = {
            "model": "phi3.5:3.8b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        }
        
        response = await provider.chat_completion(payload)
        
        # Check that the response is properly formatted
        assert response["model"] == "phi3.5:3.8b"
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert response["choices"][0]["message"]["content"] == "This is a test response from Ollama."
        assert response["choices"][0]["finish_reason"] == "stop"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 20
        assert response["usage"]["total_tokens"] == 30
        
        # Check that the correct URL was called
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://test.local:11434/api/generate"
        
        # Check that the payload was properly formatted
        sent_payload = kwargs["json"]
        assert sent_payload["model"] == "phi3.5:3.8b"
        assert "prompt" in sent_payload
        assert "<|system|>" in sent_payload["prompt"]
        assert "<|user|>" in sent_payload["prompt"]
        assert "<|assistant|>" in sent_payload["prompt"]


@pytest.mark.asyncio
async def test_format_messages():
    """Test formatting messages for Ollama."""
    provider = OllamaProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    
    formatted = provider._format_messages(messages)
    
    # Check that the messages are properly formatted
    assert "<|system|>\nYou are a helpful assistant.\n" in formatted
    assert "<|user|>\nHello!\n" in formatted
    assert "<|assistant|>\nHi there!\n" in formatted
    assert "<|user|>\nHow are you?\n" in formatted
    assert formatted.endswith("<|assistant|>\n")
    
    # Check the order of messages
    system_pos = formatted.find("<|system|>")
    user1_pos = formatted.find("<|user|>\nHello!")
    assistant_pos = formatted.find("<|assistant|>\nHi there!")
    user2_pos = formatted.find("<|user|>\nHow are you?")
    final_assistant_pos = formatted.rfind("<|assistant|>\n")
    
    assert system_pos < user1_pos < assistant_pos < user2_pos < final_assistant_pos 