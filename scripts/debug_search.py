#!/usr/bin/env python3
"""Debug script for testing search functionality."""

import asyncio
import logging
from typing import Dict, List, Any

from src.core.retriever import HybridRetriever
from src.core.reranker import JinaReranker
from src.config import settings

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("debug_search")

async def main():
    """Run search test."""
    query = "LLM agent related repositories"
    limit = 25
    
    logger.info(f"Testing search with query: '{query}', limit: {limit}")
    
    # Инициализация гибридного поиска
    retriever = HybridRetriever(
        bm25_weight=0.5,  # Равный вес для BM25
        vector_weight=0.5,  # Равный вес для векторного поиска
    )
    await retriever.initialize()
    
    # Инициализация ре-ранкера
    reranker = JinaReranker()
    await reranker._ensure_session()
    
    # Выполнение поиска
    logger.info("Executing hybrid search...")
    results = await retriever.search(query, limit=limit)
    
    # Вывод результатов
    logger.info(f"Found {len(results)} results")
    if results:
        for i, result in enumerate(results[:5]):  # Показать топ-5 результатов
            logger.info(f"Result {i+1}:")
            logger.info(f"  Repo: {result.get('repo_name', 'Unknown')}")
            logger.info(f"  URL: {result.get('repo_url', 'Unknown')}")
            logger.info(f"  Score: {result.get('score', 0)}")
            logger.info(f"  Vector Score: {result.get('vector_score', 0)}")
            logger.info(f"  BM25 Score: {result.get('bm25_score', 0)}")
            summary = result.get('summary', 'No summary')
            if summary:
                logger.info(f"  Summary: {summary[:100]}...")
            else:
                logger.info("  Summary: No summary available")
    else:
        logger.warning("No results found!")
    
    # Применение ре-ранкинга
    logger.info("Applying reranking...")
    if results and len(results) > 1:
        reranked_results = await reranker.rerank(query, results, top_k=min(limit, len(results)))
        logger.info(f"Reranked results: {len(reranked_results)}")
        
        for i, result in enumerate(reranked_results[:5]):  # Показать топ-5 результатов после ре-ранкинга
            logger.info(f"Reranked Result {i+1}:")
            logger.info(f"  Repo: {result.get('repo_name', 'Unknown')}")
            logger.info(f"  Score: {result.get('score', 0)}")
    
    # Закрытие ресурсов
    await retriever.close()
    await reranker.close()

if __name__ == "__main__":
    asyncio.run(main())