"""GitHub repository collector.

This module provides functionality to collect starred repositories from GitHub.
"""

import asyncio
import base64
import json
import logging
import re
from typing import Dict, List, Optional, Any, Set

import httpx

# Fix imports for compatibility
try:
    from src.config import settings
except ImportError:
    try:
        from config import settings
    except ImportError:
        import sys
        from pathlib import Path

        # Add the project's root directory to the module search path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.config import settings

logger = logging.getLogger(__name__)

# Constants
GITHUB_API_URL = "https://api.github.com"
GITHUB_API_REPOS = f"{GITHUB_API_URL}/users/{{}}/starred"
GITHUB_API_README = f"{GITHUB_API_URL}/repos/{{}}/readme"
MAX_CONCURRENT_REQUESTS = 10
RATE_LIMIT_WAIT_TIME = 60

# Common stop words to filter out from README content
STOP_WORDS: Set[str] = {
    "the",
    "and",
    "to",
    "for",
    "a",
    "an",
    "or",
    "in",
    "of",
    "is",
    # Additional stop words can be added here
}


class RepoCollector:
    """GitHub repository collector.

    This class handles collecting starred repositories from GitHub.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        token: Optional[str] = None,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    ) -> None:
        """Initialize the repository collector.

        Args:
            username: GitHub username (defaults to config)
            token: GitHub API token (defaults to config)
            max_concurrent: Maximum number of concurrent requests
        """
        self.username = username or settings.GITHUB_USERNAME
        self.token = token or settings.GITHUB_TOKEN
        self.semaphore = asyncio.Semaphore(max_concurrent)

        if not self.username or not self.token:
            raise ValueError("GitHub username and token must be provided")

        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        self.client = httpx.AsyncClient(timeout=30.0)
        logger.debug(f"Initialized RepoCollector for user: {self.username}")

    async def collect_starred_repos(self) -> List[Dict[str, Any]]:
        """Collect all starred repositories with their README content.

        Returns:
            List of repository data dictionaries
        """
        repos = await self._fetch_repos()

        if not repos:
            logger.warning("No repositories found")
            return []

        logger.info(f"Fetching READMEs for {len(repos)} repositories")
        tasks = [self._enrich_repo_data(repo) for repo in repos]
        return await asyncio.gather(*tasks)

    async def _fetch_repos(self) -> List[Dict[str, Any]]:
        """Fetch starred repositories with pagination.

        Returns:
            List of basic repository data
        """
        repos = []
        page = 1

        while True:
            url = GITHUB_API_REPOS.format(self.username) + f"?per_page=100&page={page}"
            response = await self._safe_request(url)

            if not response:
                break

            try:
                data = response.json()
            except json.JSONDecodeError:
                logger.error(f"JSON decode error on page {page}")
                break

            if not data:
                break

            for repo in data:
                repos.append(
                    {
                        "name": repo["full_name"],
                        "html_url": repo["html_url"],
                        "description": repo.get("description", ""),
                        "topics": repo.get("topics", []),
                        "language": repo.get("language"),
                        "stargazers_count": repo.get("stargazers_count", 0),
                        "forks_count": repo.get("forks_count", 0),
                        "created_at": repo.get("created_at"),
                        "updated_at": repo.get("updated_at"),
                    }
                )

            if len(data) < 100:
                break  # Last page reached
            page += 1

        logger.info(f"Fetched {len(repos)} repositories")
        return repos

    async def _enrich_repo_data(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich repository data with README content.

        Args:
            repo: Basic repository data

        Returns:
            Enriched repository data
        """
        readme = await self._fetch_readme(repo["name"])
        repo["readme"] = readme
        return repo

    async def _fetch_readme(self, repo_full_name: str) -> str:
        """Fetch repository README asynchronously.

        Args:
            repo_full_name: Full repository name (owner/repo)

        Returns:
            README content or empty string
        """
        url = GITHUB_API_README.format(repo_full_name)

        async with self.semaphore:
            response = await self._safe_request(url)

            if response and response.status_code == 200:
                try:
                    content = response.json().get("content", "")
                    decoded = base64.b64decode(content).decode("utf-8")
                    cleaned_readme = self._clean_text(decoded)
                    return cleaned_readme
                except Exception as e:
                    logger.warning(f"Error decoding README for {repo_full_name}: {e}")

        return ""

    async def _safe_request(self, url: str) -> Optional[httpx.Response]:
        """Handle API requests with retry logic.

        Args:
            url: API endpoint URL

        Returns:
            Response object or None
        """
        retries = 3
        for attempt in range(retries):
            try:
                response = await self.client.get(url, headers=self.headers)

                if response.status_code == 403:
                    logger.warning(
                        f"Rate limit hit. Waiting {RATE_LIMIT_WAIT_TIME}s..."
                    )
                    await asyncio.sleep(RATE_LIMIT_WAIT_TIME)
                    continue

                if response.status_code == 200:
                    return response

                logger.warning(f"Request failed with status {response.status_code}")

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e}")

            await asyncio.sleep(2**attempt)  # Exponential backoff

        return None

    def _clean_text(self, text: str) -> str:
        """Remove Markdown, HTML, links, and extra spaces.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove Markdown links
        text = re.sub(r"\[.*?\]\(.*?\)", " ", text)
        # Remove special characters
        text = re.sub(r"[^\w\s]", " ", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
