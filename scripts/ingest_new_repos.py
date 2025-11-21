#!/usr/bin/env python3
"""Incrementally ingest new repositories that aren't in Qdrant yet."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.ingestion.pipeline import IngestionPipeline
from src.core.storage import QdrantStore

console = Console()


async def get_existing_repos():
    """Get list of repositories already in Qdrant."""
    store = QdrantStore()
    await store.initialize()
    
    existing = await store.get_existing_repositories()
    await store.close()
    
    return set(existing)


async def ingest_new_repos(repos_file: Path):
    """Ingest only new repositories not yet in Qdrant."""
    console.print(f"[cyan]Loading repositories from {repos_file}...[/cyan]")
    
    # Load all repos
    all_repos = json.loads(repos_file.read_text())
    console.print(f"Found {len(all_repos)} total repositories")
    
    # Get existing repos from Qdrant
    console.print("[cyan]Checking existing repositories in Qdrant...[/cyan]")
    existing_names = await get_existing_repos()
    console.print(f"Found {len(existing_names)} existing repositories")
    
    # Filter new repos
    new_repos = [
        repo for repo in all_repos 
        if repo.get("full_name") not in existing_names
    ]
    
    console.print(f"[green]Found {len(new_repos)} new repositories to ingest[/green]")
    
    if not new_repos:
        console.print("[yellow]No new repositories to ingest![/yellow]")
        return
    
    # Ingest new repos
    pipeline = IngestionPipeline()
    await pipeline.initialize()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Ingesting {len(new_repos)} repositories...", 
                total=len(new_repos)
            )
            
            for i, repo in enumerate(new_repos, 1):
                try:
                    # Add summary if missing
                    if "summary" not in repo or not repo["summary"]:
                        enriched = await pipeline.summarizer.summarize(repo)
                        repo.update(enriched)
                    
                    # Store in Qdrant
                    await pipeline.qdrant_store.store_repositories([repo])
                    
                    progress.update(
                        task, 
                        advance=1,
                        description=f"Ingested {i}/{len(new_repos)}: {repo.get('full_name', 'unknown')}"
                    )
                    
                except Exception as e:
                    console.print(f"[red]Failed to ingest {repo.get('full_name')}: {e}[/red]")
                    progress.update(task, advance=1)
        
        console.print(f"[green]✓ Successfully ingested {len(new_repos)} new repositories![/green]")
        
    finally:
        await pipeline.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python scripts/ingest_new_repos.py <repos_file.json>[/red]")
        sys.exit(1)
    
    repos_file = Path(sys.argv[1])
    if not repos_file.exists():
        console.print(f"[red]File not found: {repos_file}[/red]")
        sys.exit(1)
    
    asyncio.run(ingest_new_repos(repos_file))
