#!/usr/bin/env python
"""Model Context Protocol (MCP) server for Oh My Repos.

Tools exposed:
- search_repos: hybrid search over indexed repositories.
- get_repo_summary: return stored summary/metadata for a repository.
- ask_about_repos: RAG-style answer using repo context.
- get_similar_repos: find related repositories based on an anchor repo.

Transport: stdio (default) for Claude Desktop / Cursor. TCP can be added later.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.config import settings
from src.core.retriever import HybridRetriever
from src.core.storage import QdrantStore
from src.llm.generator import LLMGenerator

logger = logging.getLogger(__name__)
server = Server("ohmyrepos-mcp")

_retriever: Optional[HybridRetriever] = None
_retriever_lock = asyncio.Lock()


async def get_retriever() -> HybridRetriever:
    """Lazily initialize and cache HybridRetriever."""
    global _retriever
    if _retriever is None:
        async with _retriever_lock:
            if _retriever is None:
                store: Optional[QdrantStore] = None
                if settings.qdrant:
                    store = QdrantStore()
                retriever = HybridRetriever(
                    qdrant_store=store,
                    bm25_weight=settings.search.bm25_weight,
                    vector_weight=settings.search.vector_weight,
                    bm25_variant=settings.search.bm25_variant,
                    rrf_k=settings.search.rrf_k,
                )
                await retriever.initialize()
                _retriever = retriever
                logger.info("HybridRetriever initialized for MCP server")
    return _retriever


def format_results(results: List[Dict[str, Any]], limit: int = 5) -> str:
    """Render search results into a compact text block."""
    lines = []
    for idx, repo in enumerate(results[:limit], 1):
        name = repo.get("repo_name") or repo.get("full_name") or repo.get("name", "")
        language = repo.get("language") or ""
        summary = repo.get("summary") or repo.get("description") or ""
        stars = repo.get("stars") or repo.get("stargazers_count") or ""
        tags = repo.get("tags") or repo.get("topics") or []
        tag_str = ", ".join(tags) if tags else ""
        lines.append(
            f"[{idx}] {name} ({language}) ⭐ {stars}\nSummary: {summary}\nTags: {tag_str}".strip()
        )
    return "\n\n".join(lines) if lines else "No results"


def format_context(results: List[Dict[str, Any]], limit: int = 4) -> str:
    """Build RAG context from search results."""
    context_chunks = []
    for idx, repo in enumerate(results[:limit], 1):
        name = repo.get("repo_name") or repo.get("full_name") or repo.get("name", "")
        summary = repo.get("summary") or repo.get("description") or ""
        language = repo.get("language") or ""
        tags = repo.get("tags") or repo.get("topics") or []
        context_chunks.append(
            f"[{idx}] {name}\nLanguage: {language}\nTags: {', '.join(tags) if tags else 'n/a'}\n{summary}"
        )
    return "\n\n".join(context_chunks)


def find_repo_by_name(retriever: HybridRetriever, name: str) -> Optional[Dict[str, Any]]:
    """Search local repo_data for a matching repository."""
    target = name.lower()
    for repo in retriever.repo_data:
        candidate = (
            repo.get("repo_name")
            or repo.get("full_name")
            or repo.get("name", "")
        ).lower()
        if candidate == target:
            return repo
    return None


async def tool_search_repos(arguments: Dict[str, Any]) -> List[TextContent]:
    query = (arguments.get("query") or "").strip()
    limit = int(arguments.get("limit") or 8)
    if not query:
        raise ValueError("query is required")

    retriever = await get_retriever()
    results = await retriever.search(query=query, limit=limit)
    return [TextContent(type="text", text=format_results(results, limit=limit))]


async def tool_get_repo_summary(arguments: Dict[str, Any]) -> List[TextContent]:
    name = (arguments.get("repo_full_name") or "").strip()
    if not name:
        raise ValueError("repo_full_name is required")

    retriever = await get_retriever()
    repo = find_repo_by_name(retriever, name)
    if not repo:
        # fall back to hybrid search
        results = await retriever.search(query=name, limit=1)
        repo = results[0] if results else None

    if not repo:
        return [TextContent(type="text", text=f"No repository found for '{name}'.")]

    text = json.dumps(repo, indent=2, ensure_ascii=False)
    return [TextContent(type="text", text=text)]


async def tool_get_similar_repos(arguments: Dict[str, Any]) -> List[TextContent]:
    name = (arguments.get("repo_full_name") or "").strip()
    limit = int(arguments.get("limit") or 5)
    if not name:
        raise ValueError("repo_full_name is required")

    retriever = await get_retriever()
    anchor = find_repo_by_name(retriever, name)
    if anchor:
        anchor_query = anchor.get("summary") or anchor.get("description") or name
    else:
        anchor_query = name

    results = await retriever.search(query=anchor_query, limit=limit + 1)
    filtered = [r for r in results if (r.get("repo_name") or r.get("full_name")) != name]
    return [TextContent(type="text", text=format_results(filtered, limit=limit))]


async def tool_ask_about_repos(arguments: Dict[str, Any]) -> List[TextContent]:
    question = (arguments.get("question") or "").strip()
    focus = arguments.get("repo_full_name")
    limit = int(arguments.get("limit") or 5)
    if not question:
        raise ValueError("question is required")

    retriever = await get_retriever()
    query = focus or question
    results = await retriever.search(query=query, limit=limit)
    context = format_context(results, limit=limit)

    generator = LLMGenerator(max_tokens=settings.llm.max_tokens if settings.llm else 800)
    answer = await generator.generate(prompt=question, context=context, history=None, stream=False)
    await generator.close()

    response = f"Context:\n{context}\n\nAnswer:\n{answer}"
    return [TextContent(type="text", text=response)]


@server.list_tools()
async def list_tools() -> List[Tool]:
    """Expose available tools to MCP clients."""
    return [
        Tool(
            name="search_repos",
            description="Hybrid search over indexed repositories using BM25 + vectors.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_repo_summary",
            description="Return stored metadata/summary for a repository (full name).",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_full_name": {"type": "string", "description": "owner/name"},
                },
                "required": ["repo_full_name"],
            },
        ),
        Tool(
            name="ask_about_repos",
            description="Answer a question using RAG context from indexed repositories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "repo_full_name": {"type": "string", "description": "Optional focus repository"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="get_similar_repos",
            description="Find repositories similar to the given repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_full_name": {"type": "string", "description": "owner/name to anchor similarity"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["repo_full_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[TextContent]:
    """Dispatch tool calls."""
    arguments = arguments or {}
    try:
        if name == "search_repos":
            return await tool_search_repos(arguments)
        if name == "get_repo_summary":
            return await tool_get_repo_summary(arguments)
        if name == "ask_about_repos":
            return await tool_ask_about_repos(arguments)
        if name == "get_similar_repos":
            return await tool_get_similar_repos(arguments)
    except Exception as exc:  # Return error back to client as text
        logger.error("Tool %s failed: %s", name, exc)
        return [TextContent(type="text", text=f"Error: {exc}")]

    raise ValueError(f"Unknown tool: {name}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Oh My Repos MCP server")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Ensure repos.json exists for BM25; warn otherwise
    repos_path = Path("repos.json")
    if not repos_path.exists():
        logger.warning("repos.json not found; BM25-only search will be empty until ingestion runs")

    async with stdio_server() as (read, write):
        try:
            await server.run(read, write)
        finally:
            if _retriever:
                await _retriever.close()


if __name__ == "__main__":
    asyncio.run(main())
