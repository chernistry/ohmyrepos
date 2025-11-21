import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import settings
from src.core.retriever import HybridRetriever
from src.core.storage import QdrantStore
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
    """Stream chat response with RAG."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_message = request.messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    # Perform RAG search
    context = ""
    try:
        if settings.qdrant:
            # Initialize retriever
            qdrant_store = QdrantStore()
            retriever = HybridRetriever(
                qdrant_store=qdrant_store,
                bm25_weight=settings.search.bm25_weight,
                vector_weight=settings.search.vector_weight,
                bm25_variant=settings.search.bm25_variant,
            )
            await retriever.initialize()

            # Search for relevant repositories
            results = await retriever.search(query=last_message, limit=5)

            # Format context from results
            if results:
                context_parts = []
                for r in results:
                    part = f"Repository: {r.get('repo_name')}\n"
                    part += f"Description: {r.get('summary') or r.get('description')}\n"
                    part += f"Language: {r.get('language')}\n"
                    part += f"Tags: {', '.join(r.get('tags', []))}\n"
                    context_parts.append(part)
                context = "\n---\n".join(context_parts)
                logger.info(f"RAG context found: {len(results)} repositories")
            
            await retriever.close()
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        # Continue without context

    generator = LLMGenerator()
    
    async def event_generator():
        try:
            stream = await generator.generate(
                prompt=last_message,
                context=context,
                history=history,
                stream=True
            )
            async for chunk in stream:
                yield chunk
        except Exception as e:
            logger.error(f"Error generating chat response: {e}", exc_info=True)
            yield f"Error: {str(e)}"
        finally:
            await generator.close()

    return StreamingResponse(event_generator(), media_type="text/plain")
