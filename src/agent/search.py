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

async def get_local_repos(repos_path: str) -> Set[str]:
    """Get set of local repository names from repos.json."""
    import json
    from pathlib import Path
    
    path = Path(repos_path)
    if not path.exists():
        return set()
        
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Assuming 'full_name' or 'name' is available. 
        # If 'full_name' is missing, we might need to construct it or use 'name' carefully.
        # Based on typical GitHub JSON, 'full_name' (owner/repo) is standard.
        # Let's check if our repos.json has it.
        return {r.get("full_name", r.get("name")) for r in data if r.get("name")}
    except Exception:
        return set()
