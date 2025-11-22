"""Profile analysis module for GitHub Discovery Agent.

This module analyzes the user's existing starred repositories to build an interest profile.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger("ohmyrepos.agent.profile")

class InterestCluster(BaseModel):
    """Represents a cluster of user interests."""
    name: str
    keywords: List[str]
    languages: List[str]
    score: float  # Relevance score (0.0 - 1.0)

def analyze_profile(repos_path: Path) -> List[InterestCluster]:
    """Analyze the user's starred repositories to identify interest clusters using an LLM.

    Args:
        repos_path: Path to the JSON file containing starred repositories.

    Returns:
        List of identified InterestClusters.
    """
    if not repos_path.exists():
        logger.warning(f"Repositories file not found at {repos_path}")
        return []

    try:
        data = json.loads(repos_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            logger.error("Invalid repositories file format: expected a list")
            return []
        
        logger.info(f"Analyzing profile from {len(data)} repositories...")

        # Prepare data for LLM (limit to last 100 to fit context)
        # Assuming data is sorted by something, or just take the list as is.
        # Ideally we'd sort by 'starred_at' if available, but let's just take the first 100.
        sample_repos = []
        for repo in data[:100]:
            sample_repos.append(f"- {repo.get('full_name')}: {repo.get('description')} (Language: {repo.get('language')})")
        
        repos_text = "\n".join(sample_repos)

        prompt_template = Path("prompts/profile_analysis.md").read_text(encoding="utf-8")
        prompt = prompt_template.replace("{{repos_text}}", repos_text)

        if not settings.llm:
            logger.warning("LLM not configured, falling back to empty clusters.")
            return []

        headers = {}
        if settings.llm.api_key:
            headers["Authorization"] = f"Bearer {settings.llm.api_key.get_secret_value()}"
        
        payload = {
            "model": settings.llm.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        # Synchronous call for simplicity in this synchronous function, 
        # or we could make this function async. 
        # Since the original was sync, let's keep it sync but use httpx.Client? 
        # Or better, make it async and update the caller.
        # The caller `discover` calls `asyncio.run(_discover_async)`, so `_discover_async` is async.
        # But `analyze_profile` is currently called synchronously inside `_discover_async`.
        # It's better to make `analyze_profile` async.
        
        # However, to avoid changing the signature too much right now (and breaking imports if I miss one),
        # I'll use `httpx.Client` (sync) or `httpx.run`? 
        # Actually, `httpx` has a sync `Client`.
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{settings.llm.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON
            try:
                parsed = json.loads(content)
                clusters_data = parsed.get("clusters", [])
                
                clusters = []
                for c in clusters_data:
                    clusters.append(InterestCluster(
                        name=c["name"],
                        keywords=c["keywords"],
                        languages=c["languages"],
                        score=c["score"]
                    ))
                
                # Sort by score
                clusters.sort(key=lambda x: x.score, reverse=True)
                return clusters
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response: {content}")
                return []

    except Exception as e:
        logger.exception(f"Error analyzing profile: {e}")
        return []
