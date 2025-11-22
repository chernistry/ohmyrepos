"""Profile analysis module for GitHub Discovery Agent.

This module analyzes the user's existing starred repositories to build an interest profile.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger("ohmyrepos.agent.profile")

class InterestCluster(BaseModel):
    """Represents a cluster of user interests."""
    name: str
    keywords: List[str]
    languages: List[str]
    score: float  # Relevance score based on frequency

def analyze_profile(repos_path: Path) -> List[InterestCluster]:
    """Analyze the user's starred repositories to identify interest clusters.

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

        # 1. Extract features
        all_languages = []
        all_topics = []
        
        for repo in data:
            # Language
            lang = repo.get("language")
            if lang:
                all_languages.append(lang)
            
            # Topics/Tags
            topics = repo.get("topics", []) or repo.get("tags", [])
            if isinstance(topics, list):
                all_topics.extend([t.lower() for t in topics])

        # 2. Frequency Analysis
        lang_counts = Counter(all_languages)
        topic_counts = Counter(all_topics)

        # 3. Simple Clustering (Heuristic for now)
        # In a real implementation, we might use embeddings or LLM here.
        # For now, we'll create clusters based on top languages and associated top topics.
        
        clusters = []
        
        # Get top 3 languages
        top_langs = lang_counts.most_common(3)
        
        for lang, count in top_langs:
            # Find top topics associated with this language (naive approach: global top topics)
            # A better approach would be to filter repos by language first, then count topics.
            
            lang_repos = [r for r in data if r.get("language") == lang]
            lang_topics = []
            for r in lang_repos:
                 ts = r.get("topics", []) or r.get("tags", [])
                 if isinstance(ts, list):
                     lang_topics.extend([t.lower() for t in ts])
            
            top_lang_topics = [t for t, _ in Counter(lang_topics).most_common(5)]
            
            clusters.append(
                InterestCluster(
                    name=f"{lang} Ecosystem",
                    keywords=top_lang_topics,
                    languages=[lang],
                    score=count / len(data)
                )
            )

        # Also add a "General Interests" cluster based on top global topics not covered?
        # For simplicity, let's just return the language-based clusters for V1.
        
        return clusters

    except Exception as e:
        logger.exception(f"Error analyzing profile: {e}")
        return []
