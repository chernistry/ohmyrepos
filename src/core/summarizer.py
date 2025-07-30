"""Repository summarization service.

This module provides functionality to summarize GitHub repositories using LLM.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable

# Fix imports for compatibility
try:
    from src.llm.generator import LLMGenerator
    from src.llm.prompt_builder import PromptBuilder
except ImportError:
    from llm.generator import LLMGenerator
    from llm.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class RepoSummarizer:
    """Repository summarization service.
    
    This class handles summarizing GitHub repositories using an LLM.
    """
    
    def __init__(
        self,
        generator: Optional[LLMGenerator] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        debug: bool = False,
        concurrency: int = None,
        save_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the repository summarizer.
        
        Args:
            generator: LLM generator to use
            prompt_builder: Prompt builder to use
            debug: Whether to enable debug mode
            concurrency: Maximum number of concurrent summarization tasks
                         (defaults to CPU count if None)
            save_callback: Optional callback function to save results after each repository
        """
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.debug = debug
        self.generator = generator or LLMGenerator(
            prompt_builder=self.prompt_builder, 
            debug=self.debug
        )
        
        # Set concurrency based on CPU count if not specified
        self.concurrency = concurrency or min(os.cpu_count() or 2, 4)
        self.semaphore = asyncio.Semaphore(self.concurrency)
        self.save_callback = save_callback
        self.lock = asyncio.Lock()  # Lock for thread-safe operations
        self.enriched_repos = []  # Store processed repositories
        
        logger.debug(f"Initialized RepoSummarizer with concurrency={self.concurrency}")
    
    async def summarize(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize a GitHub repository.
        
        Args:
            repo_data: Repository data containing at least description and README
            
        Returns:
            Dictionary with summary and tags
        """
        # Extract description and README
        description = repo_data.get("description", "") or ""
        readme = repo_data.get("readme", "") or ""
        
        # Log input data for debugging
        logger.debug(f"Repository name: {repo_data.get('name', 'unknown')}")
        logger.debug(f"Description length: {len(description)} chars")
        logger.debug(f"README length: {len(readme)} chars")
        
        # Truncate README if it's too long (to avoid token limits)
        original_readme_length = len(readme)
        if len(readme) > 10000:
            readme = readme[:10000] + "...[truncated]"
            logger.debug(f"Truncated README from {original_readme_length} to 10000 chars")
        
        # Generate summary
        logger.info(f"Summarizing repository: {repo_data.get('name', 'unknown')}")
        try:
            # Build prompt for debugging
            if self.debug:
                prompt = self.prompt_builder.build_summarize_repo_prompt(
                    description=description,
                    readme=readme,
                )
                logger.debug(f"Generated prompt (length: {len(prompt)} chars):\n{prompt}")
            
            # Generate summary
            result = await self.generator.generate_repo_summary(
                description=description,
                readme=readme,
            )
            
            # Log result for debugging
            logger.debug(f"Generated summary length: {len(result.get('summary', ''))}")
            logger.debug(f"Generated tags: {result.get('tags', [])}")
            
            # Ensure result has required fields
            if "summary" not in result or "tags" not in result:
                logger.warning("Generated summary is missing required fields")
                result = {
                    "summary": result.get("summary", "No summary generated"),
                    "tags": result.get("tags", []),
                }
            
            # Add original repo data
            result["repo_name"] = repo_data.get("name", "")
            result["repo_url"] = repo_data.get("html_url", "")
            
            return result
        except Exception as e:
            logger.error(f"Error summarizing repository: {e}")
            return {
                "summary": "Failed to generate summary",
                "tags": [],
                "repo_name": repo_data.get("name", ""),
                "repo_url": repo_data.get("html_url", ""),
                "error": str(e),
            }
    
    async def _summarize_with_semaphore(self, repo_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Summarize a repository with semaphore control.
        
        Args:
            repo_data: Repository data
            
        Returns:
            Tuple of (original repo data, summary result)
        """
        async with self.semaphore:
            # Remove any existing error field before processing
            if "error" in repo_data:
                repo_data_clean = repo_data.copy()
                repo_data_clean.pop("error")
                result = await self.summarize(repo_data_clean)
            else:
                result = await self.summarize(repo_data)
            
            # Update the original repo with the summary
            updated_repo = repo_data.copy()
            updated_repo.update(result)
            
            # Save the result immediately if callback is provided
            if self.save_callback:
                async with self.lock:
                    # Check if this repo is already in our list (by name)
                    repo_name = updated_repo.get("name", "unknown")
                    existing_index = None
                    
                    for i, repo in enumerate(self.enriched_repos):
                        if repo.get("name") == repo_name:
                            existing_index = i
                            break
                    
                    # Replace existing or append
                    if existing_index is not None:
                        self.enriched_repos[existing_index] = updated_repo
                    else:
                        self.enriched_repos.append(updated_repo)
                    
                    self.save_callback(self.enriched_repos)
            
            return repo_data, result
    
    async def summarize_batch(self, repos: List[Dict[str, Any]], output_file: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Summarize multiple repositories in parallel.
        
        Args:
            repos: List of repository data
            output_file: Optional path to save results incrementally
            
        Returns:
            List of repositories with summaries
        """
        logger.info(f"Summarizing {len(repos)} repositories with concurrency={self.concurrency}")
        
        # Reset enriched repos
        self.enriched_repos = []
        
        # Create a dictionary to track repositories by name to avoid duplicates
        repo_dict = {}
        
        # Setup save callback if output file is provided
        if output_file:
            def save_to_file(data):
                try:
                    output_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    logger.debug(f"Saved {len(data)} repositories to {output_file}")
                except Exception as e:
                    logger.error(f"Error saving to {output_file}: {e}")
            
            self.save_callback = save_to_file
            
            # If output file exists, load existing data to avoid duplicates
            if output_file.exists():
                try:
                    existing_data = json.loads(output_file.read_text(encoding="utf-8"))
                    if isinstance(existing_data, list):
                        logger.info(f"Loaded {len(existing_data)} existing repositories from {output_file}")
                        # Create dictionary of existing repos by name
                        for repo in existing_data:
                            name = repo.get("name")
                            if name:
                                repo_dict[name] = repo
                        logger.info(f"Found {len(repo_dict)} unique repositories in existing data")
                except Exception as e:
                    logger.error(f"Error loading existing data from {output_file}: {e}")
        else:
            self.save_callback = None
        
        # Filter out repos that already have summaries
        to_process = []
        already_processed = []
        
        for repo in repos:
            repo_name = repo.get("name", "unknown")
            
            # Check if repository is already in our dictionary from existing file
            if repo_name in repo_dict:
                existing_repo = repo_dict[repo_name]
                # Check if it already has valid summary and tags and no errors
                has_summary = existing_repo.get("summary") and isinstance(existing_repo.get("summary"), str) and len(existing_repo.get("summary", "")) > 10
                has_tags = existing_repo.get("tags") and isinstance(existing_repo.get("tags"), list) and len(existing_repo.get("tags", [])) > 0
                has_error = "error" in existing_repo
                
                if has_summary and has_tags and not has_error:
                    logger.info(f"Skipping {repo_name}: already exists in output file with valid summary and tags")
                    already_processed.append(existing_repo)
                    continue
                elif has_error:
                    logger.info(f"Reprocessing {repo_name}: previous attempt had error: {existing_repo.get('error')}")
                    # Use original repo data but keep important fields from existing data
                    merged_repo = repo.copy()
                    for key in ["html_url", "description", "readme"]:
                        if key in existing_repo and existing_repo.get(key):
                            merged_repo[key] = existing_repo.get(key)
                    to_process.append(merged_repo)
                    continue
            
            # Check if the repository already has a valid summary and tags in current data
            has_summary = repo.get("summary") and isinstance(repo.get("summary"), str) and len(repo.get("summary", "")) > 10
            has_tags = repo.get("tags") and isinstance(repo.get("tags"), list) and len(repo.get("tags", [])) > 0
            has_error = "error" in repo
            
            # Log detailed info for debugging
            logger.debug(f"Repo {repo_name}: has_summary={has_summary}, has_tags={has_tags}, has_error={has_error}")
            logger.debug(f"Summary: {repo.get('summary', 'None')[:50]}...")
            logger.debug(f"Tags: {repo.get('tags', [])}")
            
            if has_summary and has_tags and not has_error:
                logger.info(f"Skipping {repo_name}: already has valid summary and tags")
                already_processed.append(repo)
                # Update dictionary
                repo_dict[repo_name] = repo
            else:
                if has_error:
                    logger.info(f"Reprocessing {repo_name}: has error: {repo.get('error')}")
                else:
                    logger.info(f"Processing {repo_name}: missing summary or tags")
                to_process.append(repo)
                
        logger.info(f"Found {len(already_processed)} already processed repositories, processing {len(to_process)} new or error ones")
        
        # Add already processed repos to enriched repos
        self.enriched_repos = already_processed.copy()
        
        # Save initial state with already processed repos
        if self.save_callback:
            self.save_callback(self.enriched_repos)
        
        if not to_process:
            return self.enriched_repos
            
        # Process repositories in parallel with semaphore control
        tasks = [self._summarize_with_semaphore(repo) for repo in to_process]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {result}")
                continue
        
        # Final save to ensure all data is saved
        if self.save_callback:
            self.save_callback(self.enriched_repos)
            
        return self.enriched_repos
    
    async def close(self) -> None:
        """Close the generator."""
        await self.generator.close()
