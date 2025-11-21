"""Ingestion pipeline for Oh My Repos."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import httpx

from src.core.summarizer import RepoSummarizer
from src.core.storage import QdrantStore
from src.config import settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting repositories into Qdrant."""

    def __init__(
        self,
        qdrant_store: Optional[QdrantStore] = None,
        summarizer: Optional[RepoSummarizer] = None,
    ):
        self.qdrant_store = qdrant_store or QdrantStore()
        self.summarizer = summarizer or RepoSummarizer()

    async def initialize(self):
        """Initialize the pipeline."""
        await self.qdrant_store.initialize()

    async def ingest_repo(self, repo_url: str) -> Dict[str, Any]:
        """Ingest a single repository.

        Args:
            repo_url: GitHub repository URL (e.g., https://github.com/owner/repo)

        Returns:
            Ingested repository data
        """
        logger.info(f"Ingesting repository: {repo_url}")

        # Extract owner/repo from URL
        parts = repo_url.rstrip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid repository URL: {repo_url}")
        owner, repo_name = parts[-2], parts[-1]
        full_name = f"{owner}/{repo_name}"

        # Fetch metadata from GitHub API
        repo_data = await self._fetch_repo_metadata(full_name)

        # Fetch README
        readme = await self._fetch_readme(full_name)
        repo_data["readme"] = readme

        # Summarize
        enriched = await self.summarizer.summarize(repo_data)

        # Store in Qdrant
        await self.qdrant_store.store_repositories([enriched])

        logger.info(f"Successfully ingested: {full_name}")
        return enriched

    async def _fetch_repo_metadata(self, full_name: str) -> Dict[str, Any]:
        """Fetch repository metadata from GitHub API."""
        if not settings.github:
            raise ValueError("GitHub configuration not found")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.github.api_url}/repos/{full_name}",
                headers={
                    "Authorization": f"token {settings.github.token.get_secret_value()}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            response.raise_for_status()
            data = response.json()

            return {
                "name": data["name"],
                "full_name": data["full_name"],
                "description": data.get("description", ""),
                "url": data["html_url"],
                "stars": data["stargazers_count"],
                "language": data.get("language"),
                "topics": data.get("topics", []),
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
            }

    async def _fetch_readme(self, full_name: str) -> str:
        """Fetch repository README."""
        if not settings.github:
            raise ValueError("GitHub configuration not found")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.github.api_url}/repos/{full_name}/readme",
                headers={
                    "Authorization": f"token {settings.github.token.get_secret_value()}",
                    "Accept": "application/vnd.github.v3.raw",
                },
            )
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            return response.text

    async def reindex(self, repos_file: Path) -> List[Dict[str, Any]]:
        """Reindex repositories from a JSON file.

        Args:
            repos_file: Path to repos.json file

        Returns:
            List of ingested repositories
        """
        import json

        logger.info(f"Reindexing from: {repos_file}")

        # Load repositories
        repos = json.loads(repos_file.read_text())
        if not isinstance(repos, list):
            repos = [repos]

        logger.info(f"Found {len(repos)} repositories to reindex")

        # Summarize if needed
        repos_to_store = []
        for repo in repos:
            if "summary" not in repo or not repo["summary"]:
                enriched = await self.summarizer.summarize(repo)
                repos_to_store.append(enriched)
            else:
                repos_to_store.append(repo)

        # Store in Qdrant
        await self.qdrant_store.store_repositories(repos_to_store)

        logger.info(f"Successfully reindexed {len(repos_to_store)} repositories")
        return repos_to_store

    async def close(self):
        """Close the pipeline."""
        await self.qdrant_store.close()
