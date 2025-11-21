"""Enterprise-grade GitHub repository collector with comprehensive async patterns."""

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..config import GitHubConfig, settings
from ..core.models import RepositoryData


class GitHubRateLimiter:
    """Sophisticated rate limiter for GitHub API with proper backoff."""

    def __init__(self, requests_per_hour: int = 5000) -> None:
        """Initialize rate limiter.
        
        Args:
            requests_per_hour: Maximum requests per hour
        """
        self.requests_per_hour = requests_per_hour
        self.min_interval = 3600 / requests_per_hour  # seconds between requests
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire rate limit token with proper timing."""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()


class GitHubAPIError(Exception):
    """GitHub API specific error."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RepoCollector:
    """Enterprise-grade GitHub repository collector.
    
    Features:
    - Comprehensive async resource management
    - Intelligent rate limiting and backoff
    - Robust error handling and recovery
    - Memory-efficient streaming processing
    - Detailed progress tracking and logging
    - Graceful shutdown and cleanup
    """

    def __init__(
        self,
        github_config: Optional[GitHubConfig] = None,
        max_concurrent: int = 10,
        chunk_size: int = 100,
        enable_readme_fetch: bool = True,
    ) -> None:
        """Initialize the repository collector.

        Args:
            github_config: GitHub configuration
            max_concurrent: Maximum concurrent requests
            chunk_size: Batch size for processing
            enable_readme_fetch: Whether to fetch README content
        """
        self.config = github_config or settings.github
        if not self.config:
            raise ValueError("GitHub configuration is required")
        
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.enable_readme_fetch = enable_readme_fetch
        
        # Initialize components
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._rate_limiter: Optional[GitHubRateLimiter] = None
        self._initialized = False
        
        # Statistics tracking
        self.stats = {
            "repos_fetched": 0,
            "readmes_fetched": 0,
            "errors": 0,
            "start_time": 0.0,
            "end_time": 0.0,
        }

    async def __aenter__(self) -> "RepoCollector":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize collector with all required resources."""
        if self._initialized:
            return

        # Setup HTTP client with proper limits and headers
        headers = {
            "Authorization": f"token {self.config.token.get_secret_value()}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ohmyrepos/1.0.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        self._client = httpx.AsyncClient(
            base_url=str(self.config.api_url),
            headers=headers,
            timeout=httpx.Timeout(self.config.request_timeout),
            limits=httpx.Limits(
                max_connections=self.max_concurrent,
                max_keepalive_connections=5,
            ),
        )

        # Initialize concurrency controls
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._rate_limiter = GitHubRateLimiter()

        # Validate connection
        try:
            await self._validate_connection()
            self._initialized = True
        except Exception as e:
            await self.close()
            raise GitHubAPIError(f"Failed to initialize GitHub collector: {e}")

    async def _validate_connection(self) -> None:
        """Validate GitHub API connection and credentials."""
        try:
            response = await self._client.get("/user")
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise GitHubAPIError("Invalid GitHub token", 401)
            elif e.response.status_code == 403:
                raise GitHubAPIError("GitHub API rate limit exceeded", 403)
            else:
                raise GitHubAPIError(f"GitHub API error: {e}", e.response.status_code)

    async def collect_starred_repos(
        self,
        username: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> AsyncGenerator[RepositoryData, None]:
        """Collect starred repositories with streaming processing.

        Args:
            username: GitHub username (defaults to config)
            progress_callback: Optional progress callback function

        Yields:
            Repository data objects

        Raises:
            GitHubAPIError: For API-related errors
        """
        if not self._initialized:
            await self.initialize()

        username = username or self.config.username
        self.stats["start_time"] = time.time()

        try:
            # Stream repositories in chunks for memory efficiency
            async for repo_batch in self._fetch_repos_chunked(username):
                # Process batch concurrently
                if self.enable_readme_fetch:
                    tasks = [
                        self._enrich_repo_with_readme(repo) for repo in repo_batch
                    ]
                    enriched_repos = await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    enriched_repos = [
                        self._convert_to_repo_data(repo) for repo in repo_batch
                    ]

                # Yield successful results
                for repo in enriched_repos:
                    if isinstance(repo, Exception):
                        self.stats["errors"] += 1
                        continue
                    
                    self.stats["repos_fetched"] += 1
                    if progress_callback:
                        await progress_callback(self.stats["repos_fetched"])
                    
                    yield repo

        finally:
            self.stats["end_time"] = time.time()

    async def _fetch_repos_chunked(
        self, username: str
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """Fetch repositories in chunks with pagination."""
        page = 1
        
        # Use authenticated user endpoint to bypass 1000 limit
        url = "/user/starred"
        
        while True:
            params = {"per_page": self.chunk_size, "page": page}
            
            repos = await self._fetch_page_with_retry(url, params)
            
            if not repos:
                break
                
            yield repos
            
            if len(repos) < self.chunk_size:
                break  # Last page
                
            page += 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def _fetch_page_with_retry(
        self, url: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fetch a single page with retry logic."""
        async with self._semaphore:
            await self._rate_limiter.acquire()
            
            try:
                response = await self._client.get(url, params=params)
                await self._handle_api_response(response)
                return response.json()
                
            except httpx.HTTPStatusError as e:
                await self._handle_api_error(e.response)
                raise

    async def _enrich_repo_with_readme(
        self, repo: Dict[str, Any]
    ) -> RepositoryData:
        """Enrich repository data with README content."""
        repo_data = self._convert_to_repo_data(repo)
        
        try:
            readme_content = await self._fetch_readme(repo["full_name"])
            if readme_content:
                repo_data.summary = self._extract_summary_from_readme(readme_content)
                self.stats["readmes_fetched"] += 1
        except Exception:
            # Continue without README if fetch fails
            pass
            
        return repo_data

    def _convert_to_repo_data(self, repo: Dict[str, Any]) -> RepositoryData:
        """Convert GitHub API response to RepositoryData model."""
        return RepositoryData(
            repo_name=repo["full_name"],
            repo_url=repo["html_url"],
            description=repo.get("description") or "",
            tags=repo.get("topics", []),
            language=repo.get("language"),
            stars=repo.get("stargazers_count", 0),
            forks=repo.get("forks_count", 0),
            created_at=repo.get("created_at"),
            updated_at=repo.get("updated_at"),
        )

    async def _fetch_readme(self, repo_full_name: str) -> Optional[str]:
        """Fetch and decode repository README."""
        async with self._semaphore:
            await self._rate_limiter.acquire()
            
            try:
                url = f"/repos/{repo_full_name}/readme"
                response = await self._client.get(url)
                
                if response.status_code == 404:
                    return None
                    
                response.raise_for_status()
                data = response.json()
                
                # Decode base64 content
                content = data.get("content", "")
                if content:
                    decoded = base64.b64decode(content).decode("utf-8", errors="ignore")
                    return self._clean_readme_content(decoded)
                    
            except Exception:
                # Silently continue if README fetch fails
                pass
                
        return None

    def _clean_readme_content(self, content: str) -> str:
        """Clean README content for better processing."""
        import re
        
        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", content)
        # Remove Markdown links but keep text
        content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
        # Remove image references
        content = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", content)
        # Normalize whitespace
        content = re.sub(r"\s+", " ", content)
        # Remove common noise
        content = re.sub(r"#+\s*", "", content)  # Headers
        content = re.sub(r"\*+\s*", "", content)  # Lists
        
        return content.strip()

    def _extract_summary_from_readme(self, readme: str) -> str:
        """Extract a concise summary from README content."""
        # Take first substantial paragraph (50-500 chars)
        paragraphs = [p.strip() for p in readme.split("\n\n") if p.strip()]
        
        for paragraph in paragraphs[:3]:  # Check first 3 paragraphs
            if 50 <= len(paragraph) <= 500:
                return paragraph
                
        # Fallback: take first 200 chars
        if readme:
            return readme[:200].strip() + "..." if len(readme) > 200 else readme
            
        return ""

    async def _handle_api_response(self, response: httpx.Response) -> None:
        """Handle API response with rate limit awareness."""
        if response.is_success:
            return

        # Handle rate limiting
        if response.status_code == 403:
            if "rate limit" in response.text.lower():
                reset_time = response.headers.get("X-RateLimit-Reset")
                if reset_time:
                    wait_time = int(reset_time) - int(time.time())
                    if wait_time > 0:
                        await asyncio.sleep(min(wait_time, 3600))  # Max 1 hour wait
                raise GitHubAPIError("GitHub API rate limit exceeded", 403)
                
        response.raise_for_status()

    async def _handle_api_error(self, response: httpx.Response) -> None:
        """Handle API errors with proper classification."""
        status_code = response.status_code
        
        try:
            error_data = response.json()
            message = error_data.get("message", "Unknown error")
        except (json.JSONDecodeError, KeyError):
            message = f"HTTP {status_code}: {response.text}"

        if status_code == 401:
            raise GitHubAPIError(f"Authentication error: {message}", status_code)
        elif status_code == 403:
            raise GitHubAPIError(f"Permission error: {message}", status_code)
        elif status_code == 404:
            raise GitHubAPIError(f"Resource not found: {message}", status_code)
        else:
            raise GitHubAPIError(f"GitHub API error: {message}", status_code)

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        return {
            **self.stats,
            "duration_seconds": duration,
            "repos_per_second": (
                self.stats["repos_fetched"] / duration if duration > 0 else 0
            ),
            "success_rate": (
                (self.stats["repos_fetched"] / 
                 (self.stats["repos_fetched"] + self.stats["errors"]))
                if (self.stats["repos_fetched"] + self.stats["errors"]) > 0 else 0
            ),
        }

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        
        self._semaphore = None
        self._rate_limiter = None
        self._initialized = False


@asynccontextmanager
async def github_collector(
    github_config: Optional[GitHubConfig] = None,
    **kwargs: Any,
) -> AsyncGenerator[RepoCollector, None]:
    """Context manager for GitHub repository collector.
    
    Args:
        github_config: GitHub configuration
        **kwargs: Additional collector arguments
        
    Yields:
        Initialized RepoCollector instance
    """
    collector = RepoCollector(github_config=github_config, **kwargs)
    try:
        async with collector:
            yield collector
    finally:
        await collector.close()