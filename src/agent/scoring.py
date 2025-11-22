"""Scoring module for GitHub Discovery Agent.

This module handles fetching READMEs and scoring repositories using LLM.
"""

import logging
import json
from typing import List, Optional
import base64

import httpx
from pydantic import BaseModel

from src.config import settings
from src.agent.search import RepoMetadata
from src.agent.profile import InterestCluster

logger = logging.getLogger("ohmyrepos.agent.scoring")

class ScoredRepo(BaseModel):
    """Repository with quality score and reasoning."""
    repo: RepoMetadata
    score: float
    reasoning: str
    readme_summary: Optional[str] = None

async def fetch_readme(repo_full_name: str) -> str:
    """Fetch README content from GitHub."""
    url = f"https://api.github.com/repos/{repo_full_name}/readme"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "OhMyRepos-Discovery-Agent",
    }
    
    if settings.github and settings.github.token:
        headers["Authorization"] = f"Bearer {settings.github.token.get_secret_value()}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10.0)
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            data = response.json()
            content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
            return content[:5000]  # Truncate to avoid context limit issues
        except Exception as e:
            logger.warning(f"Failed to fetch README for {repo_full_name}: {e}")
            return ""

async def score_candidates(
    candidates: List[RepoMetadata],
    profile: InterestCluster
) -> List[ScoredRepo]:
    """Score candidate repositories against the user profile."""
    scored_repos = []
    
    for repo in candidates:
        readme = await fetch_readme(repo.full_name)
        
        from pathlib import Path
        prompt_template = Path("prompts/repo_scoring.md").read_text(encoding="utf-8")
        prompt = prompt_template.replace("{{profile_name}}", profile.name) \
            .replace("{{profile_keywords}}", ', '.join(profile.keywords)) \
            .replace("{{repo_full_name}}", repo.full_name) \
            .replace("{{repo_description}}", str(repo.description)) \
            .replace("{{repo_language}}", str(repo.language)) \
            .replace("{{repo_stars}}", str(repo.stars)) \
            .replace("{{readme_content}}", readme[:5000])
        
        try:
            # Use the configured LLM
            if not settings.llm:
                logger.warning("LLM not configured, skipping scoring.")
                continue

            headers = {}
            if settings.llm.api_key:
                headers["Authorization"] = f"Bearer {settings.llm.api_key.get_secret_value()}"
            
            payload = {
                "model": settings.llm.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.llm.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON from content (handle potential markdown code blocks)
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                data = json.loads(content.strip())
                
                scored_repo = ScoredRepo(
                    repo=repo,
                    score=float(data["score"]),
                    reasoning=data["reasoning"],
                    readme_summary=data.get("summary")
                )
                scored_repos.append(scored_repo)
                
        except Exception as e:
            logger.error(f"Failed to score {repo.full_name}: {e}")
            # Fallback: add with neutral score if LLM fails? Or skip?
            # Let's skip for now to ensure high quality.
            continue
            
    # Sort by score descending
    scored_repos.sort(key=lambda x: x.score, reverse=True)
    return scored_repos
