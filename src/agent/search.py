"""GitHub Search module for Discovery Agent.

This module handles searching GitHub for repositories and deduplicating results.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta

import httpx
from pydantic import BaseModel

from src.config import settings
from src.core.storage import QdrantStore

logger = logging.getLogger("ohmyrepos.agent.search")

class RepoMetadata(BaseModel):
    """Minimal metadata for a discovered repository."""
    name: str
    full_name: str
    url: str
    description: Optional[str]
    stars: int
    language: Optional[str]
    updated_at: str
    topics: List[str]

async def search_github(
    query: str,
    min_stars: int = 100,
    max_results: int = 20,
    excluded_repos: Optional[Set[str]] = None
) -> List[RepoMetadata]:
    """Search GitHub for repositories matching the query.

    Args:
        query: Search query string.
        min_stars: Minimum number of stars.
        max_results: Maximum number of results to return.
        excluded_repos: Set of repository names (full_name) to exclude.

    Returns:
        List of RepoMetadata objects.
    """
    if excluded_repos is None:
        excluded_repos = set()

    # Construct search query
    # Ensure we filter for public repos and apply star count
    search_query = f"{query} stars:>={min_stars} is:public"
    
    # Filter by pushed date (e.g., last year) to ensure freshness
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    search_query += f" pushed:>{one_year_ago}"

    logger.info(f"Searching GitHub with query: '{search_query}'")

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "OhMyRepos-Discovery-Agent",
    }
    
    if settings.github and settings.github.token:
        headers["Authorization"] = f"Bearer {settings.github.token.get_secret_value()}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": search_query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": min(100, max_results * 2) # Fetch more to allow for dedup
                },
                headers=headers,
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            results = []
            
            for item in items:
                full_name = item.get("full_name")
                
                # Deduplication
                if full_name in excluded_repos:
                    continue
                
                # Create metadata object
                repo = RepoMetadata(
                    name=item.get("name"),
                    full_name=full_name,
                    url=item.get("html_url"),
                    description=item.get("description"),
                    stars=item.get("stargazers_count"),
                    language=item.get("language"),
                    updated_at=item.get("updated_at"),
                    topics=item.get("topics", [])
                )
                results.append(repo)
                
                if len(results) >= max_results:
                    break
            
            logger.info(f"Found {len(results)} new repositories after deduplication")
            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"GitHub API error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logger.exception(f"Search failed: {e}")
            return []

async def generate_search_queries(intent: str) -> List[str]:
    """Generate optimized GitHub search queries from a natural language intent using LLM.

    Args:
        intent: The user's search intent (e.g., "Find me RAG frameworks").

    Returns:
        List of GitHub search query strings.
    """
    if not settings.llm:
        logger.warning("LLM not configured, falling back to simple query.")
        return [intent]

    prompt = f"""
    You are an expert at searching GitHub. Convert the following user intent into 3-5 distinct, optimized GitHub search queries to find relevant repositories.
    
    User Intent: "{intent}"
    
    Rules:
    1. Use specific GitHub search qualifiers like `topic:`, `language:`, `description:`.
    2. Vary the keywords to cover different aspects (e.g., synonyms, related technologies).
    3. Do NOT include `stars:>=` or `pushed:>` qualifiers as these are handled programmatically.
    4. Return ONLY a JSON object with a list of strings.

    Example Output:
    {{
        "queries": [
            "topic:rag language:python",
            "retrieval augmented generation description:framework",
            "topic:llm-agent topic:orchestration"
        ]
    }}
    """

    headers = {}
    if settings.llm.api_key:
        headers["Authorization"] = f"Bearer {settings.llm.api_key.get_secret_value()}"
    
    payload = {
        "model": settings.llm.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{settings.llm.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=10.0
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            data = json.loads(content)
            return data.get("queries", [intent])
            
        except Exception as e:
            logger.error(f"Failed to generate search queries: {e}")
            return [intent]

async def smart_search(
    intent: str,
    min_stars: int = 100,
    max_results: int = 20,
    excluded_repos: Optional[Set[str]] = None
) -> List[RepoMetadata]:
    """Perform a smart search using multiple generated queries.

    Args:
        intent: User's natural language intent.
        min_stars: Minimum stars.
        max_results: Maximum total results.
        excluded_repos: Set of repos to exclude.

    Returns:
        List of unique RepoMetadata objects.
    """
    if excluded_repos is None:
        excluded_repos = set()

    # 1. Generate Queries
    queries = await generate_search_queries(intent)
    logger.info(f"Generated queries for '{intent}': {queries}")

    # 2. Run searches in parallel
    tasks = []
    for query in queries:
        tasks.append(
            search_github(
                query=query,
                min_stars=min_stars,
                max_results=max_results, # Fetch max per query to ensure enough candidates
                excluded_repos=excluded_repos
            )
        )
    
    results_list = await asyncio.gather(*tasks)
    
    # 3. Aggregate and Deduplicate
    all_results = []
    seen_full_names = set(excluded_repos)
    
    for batch in results_list:
        for repo in batch:
            if repo.full_name not in seen_full_names:
                all_results.append(repo)
                seen_full_names.add(repo.full_name)
    
    # 4. Sort by stars (or maybe mix them?)
    # Let's sort by stars for now to surface highest quality first
    all_results.sort(key=lambda x: x.stars, reverse=True)
    
    return all_results[:max_results]

async def get_local_repos(repos_path: str) -> Set[str]:
    """Get set of local repository names from repos.json."""
    import json
    from pathlib import Path
    
    path = Path(repos_path)
    if not path.exists():
        return set()
        
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {r.get("full_name", r.get("name")) for r in data if r.get("name")}
    except Exception:
        return set()
