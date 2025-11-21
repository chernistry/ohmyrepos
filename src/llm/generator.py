"""LLM generator for Oh My Repos.

This module provides a generator for the Oh My Repos application,
which uses an LLM to generate repository summaries.
"""

import json
import logging
from typing import Dict, Optional, Any, Union, AsyncGenerator, List

from src.llm.chat_adapter import ChatAdapter
from src.llm.prompt_builder import PromptBuilder
from src.llm.reply_extractor import extract_json_dict_sync

logger = logging.getLogger(__name__)


class LLMGenerator:
    """LLM generator for repository summarization.

    This class handles generating structured summaries using an LLM.
    """

    def __init__(
        self,
        chat_adapter: Optional[ChatAdapter] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        max_tokens: int = 1024,
        debug: bool = False,
    ) -> None:
        """Initialize the LLM generator.

        Args:
            chat_adapter: Chat adapter to use
            prompt_builder: Prompt builder to use
            max_tokens: Maximum tokens to generate
            debug: Whether to enable debug mode
        """
        self.chat_adapter = chat_adapter or ChatAdapter()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.max_tokens = max_tokens
        self.debug = debug

        logger.debug(
            "Initialized LLMGenerator with max_tokens=%d",
            self.max_tokens,
        )

    async def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate a response based on the prompt.

        Args:
            prompt: Prompt text (user question)
            context: Optional RAG context
            history: Optional chat history (list of dicts with role and content)
            stream: Whether to stream the response

        Returns:
            Generated response text or streaming generator
        """
        # Build system message
        if context:
            system_message = self.prompt_builder.build_rag_system_message()
        else:
            system_message = self.prompt_builder.build_system_message()

        # Log system message for debugging
        if self.debug:
            logger.debug(f"System message: {system_message}")

        # Prepare messages list
        messages = [{"role": "system", "content": system_message}]
        
        # Add history if provided
        if history:
            messages.extend(history)
            
        # Prepare user content
        if context:
            user_content = self.prompt_builder.build_rag_user_prompt(prompt, context)
        else:
            user_content = prompt
            
        messages.append({"role": "user", "content": user_content})

        # Prepare payload
        payload = {
            "messages": messages,
            "temperature": 0.2,  # Lower temperature for more deterministic outputs
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

        # Log payload for debugging
        if self.debug:
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        # Generate response
        logger.info("Generating response for prompt")
        response = await self.chat_adapter.chat_completion(payload)

        if not stream:
            # Extract content from response
            content = self._extract_content(response)

            # Log raw response for debugging
            if self.debug:
                logger.debug(f"Raw LLM response: {content}")

            return content
        else:
            # Return streaming generator
            return self._stream_content(response)  # type: ignore

    async def generate_repo_summary(
        self, description: str, readme: str
    ) -> Dict[str, Any]:
        """Generate a repository summary.

        Args:
            description: Repository description
            readme: Repository README content

        Returns:
            Dictionary with summary and tags
        """
        # Build prompt
        prompt = self.prompt_builder.build_summarize_repo_prompt(
            description=description,
            readme=readme,
        )

        # Log prompt length for debugging
        logger.debug(f"Prompt length: {len(prompt)} characters")

        # Generate response
        start_time = None
        if self.debug:
            import time

            start_time = time.time()

        response_text = await self.generate(prompt, stream=False)

        # Log timing information for debugging
        if self.debug and start_time:
            import time

            elapsed = time.time() - start_time
            logger.debug(f"LLM response time: {elapsed:.2f} seconds")
            logger.debug(f"Response length: {len(response_text)} characters")

        # Extract JSON from response
        result = extract_json_dict_sync(response_text)

        if not result:
            logger.error("Failed to extract JSON from response")
            logger.debug(f"Raw response: {response_text}")
            return {"summary": "Failed to generate summary", "tags": []}

        return result

    async def close(self) -> None:
        """Close the chat adapter."""
        await self.chat_adapter.close()

    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract content from response.

        Args:
            response: Response from chat completion

        Returns:
            Extracted content
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting content from response: {e}")
            if self.debug:
                logger.debug(f"Raw response: {json.dumps(response, indent=2)}")
            return ""

    async def _stream_content(
        self, stream_gen: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Process streaming content.

        Args:
            stream_gen: Stream generator from chat completion

        Yields:
            Processed content chunks
        """
        logger.debug("Starting stream processing")
        chunk_count = 0
        async for chunk in stream_gen:
            chunk_count += 1
            try:
                # Handle strongly-typed streaming chunks
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            logger.debug(f"Yielding content chunk {chunk_count}: {len(content)} chars")
                            yield content
                    continue

                # Fallback: raw SSE line strings
                if isinstance(chunk, str):
                    if chunk.startswith("data: "):
                        data = chunk[6:].strip()
                        if data and data != "[DONE]":
                            data_json = json.loads(data)
                            delta = data_json["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                    elif chunk.strip():
                        yield chunk
            except Exception as e:
                logger.error(f"Error processing stream chunk: {e}")
                if self.debug:
                    logger.debug(f"Raw chunk: {chunk}")
                continue
        logger.debug(f"Stream processing complete. Total chunks: {chunk_count}")
