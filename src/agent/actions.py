"""Actions module for GitHub Discovery Agent.

This module handles user actions on discovered repositories, such as starring on GitHub
and ingesting into the knowledge base.
"""

import logging
import asyncio
from typing import Optional, Dict, Any

import httpx

from src.config import settings
from src.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger("ohmyrepos.agent.actions")

class ActionManager:
    """Manages actions on repositories."""

    def __init__(self):
        self.ingestion_pipeline = IngestionPipeline()

    async def star_repo(self, repo_full_name: str) -> bool:
        """Star a repository on GitHub.

        Args:
            repo_full_name: The full name of the repository (owner/repo).

        Returns:
            True if successful, False otherwise.
        """
        if not settings.github or not settings.github.token:
            logger.warning("GitHub token not configured. Cannot star repository.")
            return False

        url = f"{settings.github.api_url}/user/starred/{repo_full_name}"
        headers = {
            "Authorization": f"Bearer {settings.github.token.get_secret_value()}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Length": "0",
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(url, headers=headers, timeout=10.0)
                if response.status_code == 204:
                    logger.info(f"[ACTION] Starred repo: {repo_full_name}")
                    return True
                else:
                    logger.error(f"Failed to star {repo_full_name}: {response.status_code} - {response.text}")
                    return False
            except Exception as e:
                logger.exception(f"Error starring {repo_full_name}: {e}")
                return False

    async def ingest_repo(self, repo_url: str) -> bool:
        """Ingest a repository into the knowledge base.

        Args:
            repo_url: The URL of the repository.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Initialize pipeline if needed (it might be lightweight)
            # The pipeline uses QdrantStore which needs initialization
            await self.ingestion_pipeline.initialize()
            
            result = await self.ingestion_pipeline.ingest_repo(repo_url)
            
            if result:
                logger.info(f"[ACTION] Ingested repo: {repo_url}")
                return True
            else:
                # ingest_repo returns None if skipped (up to date), which we can consider success or "no-op"
                # But if it failed, it likely raised an exception.
                # Let's assume None means "already ingested/skipped" which is fine.
                logger.info(f"[ACTION] Repo already ingested or skipped: {repo_url}")
                return True
                
        except Exception as e:
            logger.exception(f"Error ingesting {repo_url}: {e}")
            return False
        finally:
            # We should probably close the pipeline to release resources if it's a one-off action
            # But if we reuse ActionManager, maybe we keep it open?
            # For CLI usage, we can close it.
            await self.ingestion_pipeline.close()
