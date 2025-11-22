import asyncio
import logging
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

from src.agent.profile import analyze_profile, InterestCluster
from src.agent.search import search_github, get_local_repos
from src.core.storage import QdrantStore

logger = logging.getLogger("ohmyrepos.agent.discovery")
console = Console()

def discover(
    repos_path: Path,
    max_results: int = 20,
    category: Optional[str] = None
):
    """Run the discovery pipeline.

    Args:
        repos_path: Path to the local repositories JSON file.
        max_results: Maximum number of repositories to recommend.
        category: Optional specific category to focus on.
    """
    asyncio.run(_discover_async(repos_path, max_results, category))

async def _discover_async(
    repos_path: Path,
    max_results: int = 20,
    category: Optional[str] = None
):
    console.print("[bold blue]GitHub Discovery Agent[/bold blue]")
    console.print(f"Analyzing profile from: {repos_path}")

    # Stage 1: Profile Analysis
    clusters = analyze_profile(repos_path)
    
    if not clusters:
        console.print("[yellow]No interest clusters found. Try starring more repos![/yellow]")
        return

    console.print("\n[bold]Identified Interest Clusters:[/bold]")
    _print_clusters(clusters)

    # Stage 2: Selection
    selected_query = category
    if not selected_query:
        # Interactive selection
        choices = [c.name for c in clusters] + ["Custom Query"]
        
        console.print("\n[bold]Select an interest to explore:[/bold]")
        for idx, choice in enumerate(choices, 1):
            console.print(f"{idx}. {choice}")
            
        choice_idx = int(Prompt.ask("Enter number", default="1")) - 1
        
        if 0 <= choice_idx < len(clusters):
            cluster = clusters[choice_idx]
            # Construct query from cluster keywords
            selected_query = f"language:{cluster.languages[0]} " + " ".join(cluster.keywords[:3])
            console.print(f"[green]Selected cluster: {cluster.name}[/green]")
        else:
            selected_query = Prompt.ask("Enter your search query")

    console.print(f"\n[bold]Searching GitHub for:[/bold] '{selected_query}'")

    # Stage 3: Search & Dedup
    # Load local repos to exclude
    local_repos = await get_local_repos(repos_path)
    
    # Load Qdrant repos to exclude
    store = QdrantStore()
    try:
        qdrant_repos = await store.get_existing_repositories()
    except Exception as e:
        logger.warning(f"Could not fetch Qdrant repos: {e}")
        qdrant_repos = set()
    
    excluded = local_repos.union(qdrant_repos)
    console.print(f"Excluding {len(excluded)} known repositories.")

    candidates = await search_github(
        query=selected_query,
        max_results=max_results,
        excluded_repos=excluded
    )

    if not candidates:
        console.print("[yellow]No new repositories found matching your criteria.[/yellow]")
        return

    console.print(f"\n[bold green]Found {len(candidates)} candidates:[/bold green]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Stars")
    table.add_column("Language")
    table.add_column("Description")
    
    for repo in candidates:
        table.add_row(
            f"[link={repo.url}]{repo.full_name}[/link]",
            str(repo.stars),
            repo.language or "N/A",
            (repo.description or "")[:50] + "..."
        )
    
    console.print(table)
    
    # Placeholder for future stages
    console.print("\n[dim]Scoring stage is not yet implemented.[/dim]")

def _print_clusters(clusters: List[InterestCluster]):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Cluster Name")
    table.add_column("Keywords")
    table.add_column("Languages")
    table.add_column("Score")

    for cluster in clusters:
        table.add_row(
            cluster.name,
            ", ".join(cluster.keywords),
            ", ".join(cluster.languages),
            f"{cluster.score:.2f}"
        )
    console.print(table)
