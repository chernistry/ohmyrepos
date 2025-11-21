import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.llm.generator import LLMGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@router.post("/chat")
async def chat(request: ChatRequest):
    """Stream chat response."""
    generator = LLMGenerator()
    
    # Convert messages to format expected by LLMGenerator (if needed)
    # For now, we'll just use the last user message as prompt, 
    # but ideally LLMGenerator should support full history.
    # Looking at LLMGenerator.generate, it takes a prompt string.
    # We'll construct a prompt from history or just use the last message.
    
    last_message = request.messages[-1].content if request.messages else ""
    
    # TODO: Enhance LLMGenerator to accept message history
    
    async def event_generator():
        try:
            stream = await generator.generate(last_message, stream=True)
            async for chunk in stream:
                yield chunk
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
