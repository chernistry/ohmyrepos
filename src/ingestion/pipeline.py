"""Ingestion pipeline for Oh My Repos."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
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

    async def should_reindex(self, repo_id: str, github_pushed_at: str) -> bool:
        """Check if repository needs reindexing.
        
        Args:
            repo_id: Repository ID (owner/name)
            github_pushed_at: Last push timestamp from GitHub
            
        Returns:
            True if reindexing is needed
        """
        try:
            # Retrieve existing point
            # Note: This assumes QdrantStore has a method to get a point by ID
            # If not, we might need to add it or use search with filter
            # For now, we'll assume we can search by ID
            points = await self.qdrant_store.client.retrieve(
                collection_name=self.qdrant_store.collection_name,
                ids=[self.qdrant_store._generate_id(repo_id)]
            )
            
            if not points:
                return True
                
            payload = points[0].payload
            stored_pushed_at = payload.get("last_pushed_at")
            
            if not stored_pushed_at:
                return True
                
            # Compare dates
            # GitHub format: 2011-01-26T19:01:12Z
            gh_date = datetime.fromisoformat(github_pushed_at.replace("Z", "+00:00"))
            stored_date = datetime.fromisoformat(stored_pushed_at.replace("Z", "+00:00"))
            
            return gh_date > stored_date
            
        except Exception as e:
            logger.warning(f"Error checking reindex status for {repo_id}: {e}")
            return True

    async def ingest_repo(self, repo_url: str, force: bool = False) -> Optional[Dict[str, Any]]:
        """Ingest a single repository.

        Args:
            repo_url: GitHub repository URL (e.g., https://github.com/owner/repo)
            force: Force reindexing even if up to date

        Returns:
            Ingested repository data or None if skipped
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
        
        # Check if we need to reindex
        if not force and not await self.should_reindex(full_name, repo_data["updated_at"]):
            logger.info(f"Skipping {full_name} (up to date)")
            return None

        # Fetch README
        readme = await self._fetch_readme(full_name)
        repo_data["readme"] = readme

        # Summarize
        enriched = await self.summarizer.summarize(repo_data)
        
        # Add tracking metadata
        enriched["last_pushed_at"] = repo_data["updated_at"]
        enriched["last_indexed_at"] = datetime.utcnow().isoformat() + "Z"

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
                "updated_at": data["updated_at"], # This is usually pushed_at or updated_at
                "pushed_at": data.get("pushed_at"), # Prefer pushed_at for code changes
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

    async def reindex(self, repos_file: Path, incremental: bool = False) -> List[Dict[str, Any]]:
        """Reindex repositories from a JSON file.

        Args:
            repos_file: Path to repos.json file
            incremental: If True, only index repos not already in Qdrant

        Returns:
            List of ingested repositories
        """
        import json

        logger.info(f"Reindexing from: {repos_file}")

        # Load repositories
        repos = json.loads(repos_file.read_text())
        if not isinstance(repos, list):
            repos = [repos]

        total_loaded = len(repos)
        logger.info(f"Loaded {total_loaded} repositories from file")

        # Filter out existing repos if incremental
        if incremental:
            logger.info("Checking existing repositories in Qdrant...")
            existing = await self.qdrant_store.get_existing_repositories()
            existing_names = set(existing)
            logger.info(f"Found {len(existing_names)} existing repositories in Qdrant")
            
            repos = [r for r in repos if r.get("full_name") not in existing_names]
            logger.info(f"Filtered to {len(repos)} new repositories (skipped {total_loaded - len(repos)} existing)")

        if not repos:
            logger.info("No new repositories to index")
            return []

        # Summarize if needed
        repos_to_store = []
        total_to_process = len(repos)
        
        for idx, repo in enumerate(repos, 1):
            repo_name = repo.get("full_name", "unknown")
            logger.info(f"Processing {idx}/{total_to_process}: {repo_name}")
            
            if "summary" not in repo or not repo["summary"]:
                logger.info(f"  → Generating summary for {repo_name}")
                enriched = await self.summarizer.summarize(repo)
                repos_to_store.append(enriched)
            else:
                logger.info(f"  → Using existing summary for {repo_name}")
                repos_to_store.append(repo)

        # Store in Qdrant
        logger.info(f"Storing {len(repos_to_store)} repositories in Qdrant...")
        await self.qdrant_store.store_repositories(repos_to_store)

        logger.info(f"✓ Successfully indexed {len(repos_to_store)} repositories")
        return repos_to_store

    async def close(self):
        """Close the pipeline."""
        await self.qdrant_store.close()
