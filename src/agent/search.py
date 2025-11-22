"""GitHub Search module for Discovery Agent.

This module handles searching GitHub for repositories and deduplicating results.
"""

import asyncio
import logging
import json
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
    excluded_repos: Optional[Set[str]] = None,
    strict_time: bool = True
) -> List[RepoMetadata]:
    """Search GitHub for repositories matching the query.

    Args:
        query: Search query string.
        min_stars: Minimum number of stars.
        max_results: Maximum number of results to return.
        excluded_repos: Set of repository names (full_name) to exclude.
        strict_time: If True, filter by recent activity (last year).

    Returns:
        List of RepoMetadata objects.
    """
    if excluded_repos is None:
        excluded_repos = set()

    # Construct search query
    # Ensure we filter for public repos and apply star count
    search_query = f"{query} stars:>={min_stars} is:public"
    
    # Filter by pushed date (e.g., last year) to ensure freshness
    if strict_time:
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

async def generate_search_queries(intent: str, strategy: str = "specific") -> List[str]:
    """Generate optimized GitHub search queries from a natural language intent using LLM.

    Args:
        intent: The user's search intent (e.g., "Find me RAG frameworks").
        strategy: "specific" for precise queries, "broad" for wider discovery.

    Returns:
        List of GitHub search query strings.
    """
    if not settings.llm:
        logger.warning("LLM not configured, falling back to simple query.")
        return [intent]

    from pathlib import Path
    prompt_template = Path("prompts/search_query_generation.md").read_text(encoding="utf-8")
    prompt = prompt_template.replace("{{intent}}", intent).replace("{{strategy}}", strategy)

    headers = {}
    if settings.llm.api_key:
        headers["Authorization"] = f"Bearer {settings.llm.api_key.get_secret_value()}"
    
    payload = {
        "model": settings.llm.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4 if strategy == "broad" else 0.2,
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

async def expand_domain_terms(intent: str) -> List[str]:
    """Ask LLM for technical terms and libraries related to the intent."""
    if not settings.llm:
        return []

    prompt = f"""
    Task: List 5-10 specific technical terms, libraries, or frameworks related to the user's intent.
    Intent: "{intent}"
    
    Output JSON: {{ "terms": ["term1", "term2"] }}
    """
    
    headers = {}
    if settings.llm.api_key:
        headers["Authorization"] = f"Bearer {settings.llm.api_key.get_secret_value()}"
    
    payload = {
        "model": settings.llm.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "response_format": {"type": "json_object"}
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{settings.llm.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
            content = json.loads(data["choices"][0]["message"]["content"])
            return content.get("terms", [])
        except Exception as e:
            logger.warning(f"Failed to expand domain terms: {e}")
            return []

async def smart_search(
    intent: str,
    min_stars: int = 100,
    max_results: int = 20,
    excluded_repos: Optional[Set[str]] = None
) -> List[RepoMetadata]:
    """Perform a smart search using multiple generated queries with iterative relaxation.

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

    all_results = []
    seen_full_names = set(excluded_repos)
    
    # 0. Cognitive Improvement: Expand domain terms
    domain_terms = await expand_domain_terms(intent)
    if domain_terms:
        logger.info(f"Expanded domain terms: {domain_terms}")
        # Enrich intent with top 3 terms for better context
        enriched_intent = f"{intent} (keywords: {', '.join(domain_terms[:3])})"
    else:
        enriched_intent = intent

    # Strategy 1: Specific queries with strict constraints
    logger.info(f"Deep Search [1/3]: Specific queries for '{enriched_intent}'")
    queries = await generate_search_queries(enriched_intent, strategy="specific")
    
    # Initial search with strict constraints
    results = await _execute_search_batch(queries, min_stars, max_results, seen_full_names, strict_time=True)
    all_results.extend(results)
    
    if len(all_results) >= max_results // 2:
        return all_results[:max_results]
        
    # Strategy 2: Relax time constraint and star count
    logger.info(f"Deep Search [2/3]: Relaxing constraints (time & stars) for '{enriched_intent}'")
    # Use same queries but relax constraints in execution
    results = await _execute_search_batch(queries, min_stars // 2, max_results, seen_full_names, strict_time=False)
    all_results.extend(results)
    
    if len(all_results) >= max_results // 2:
        return all_results[:max_results]

    # Strategy 3: Broad queries
    logger.info(f"Deep Search [3/3]: Generating broad queries for '{enriched_intent}'")
    broad_queries = await generate_search_queries(enriched_intent, strategy="broad")
    results = await _execute_search_batch(broad_queries, min_stars // 4, max_results, seen_full_names, strict_time=False)
    all_results.extend(results)

    # Sort by stars
    all_results.sort(key=lambda x: x.stars, reverse=True)
    
    return all_results[:max_results]

async def _execute_search_batch(
    queries: List[str],
    min_stars: int,
    max_results: int,
    excluded_repos: Set[str],
    strict_time: bool
) -> List[RepoMetadata]:
    """Helper to execute a batch of queries."""
    tasks = []
    for query in queries:
        tasks.append(
            search_github(
                query=query,
                min_stars=min_stars,
                max_results=max_results,
                excluded_repos=excluded_repos,
                strict_time=strict_time
            )
        )
    
    results_list = await asyncio.gather(*tasks)
    
    batch_results = []
    for batch in results_list:
        for repo in batch:
            if repo.full_name not in excluded_repos:
                batch_results.append(repo)
                excluded_repos.add(repo.full_name)
                
    return batch_results

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
