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
from src.agent.scoring import score_candidates
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
    selected_cluster = None
    
    if not selected_query:
        # Interactive selection
        choices = [c.name for c in clusters] + ["Custom Query"]
        
        console.print("\n[bold]Select an interest to explore:[/bold]")
        for idx, choice in enumerate(choices, 1):
            console.print(f"{idx}. {choice}")
            
        choice_idx = int(Prompt.ask("Enter number", default="1")) - 1
        
        if 0 <= choice_idx < len(clusters):
            selected_cluster = clusters[choice_idx]
            # Construct query from cluster keywords
            selected_query = f"language:{selected_cluster.languages[0]} " + " ".join(selected_cluster.keywords[:3])
            console.print(f"[green]Selected cluster: {selected_cluster.name}[/green]")
        else:
            selected_query = Prompt.ask("Enter your search query")
            # Create a dummy cluster for custom query
            selected_cluster = InterestCluster(name="Custom", keywords=selected_query.split(), languages=[], score=1.0)

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

    console.print(f"\n[bold]Scoring {len(candidates)} candidates...[/bold]")

    # Stage 4: Scoring
    scored_results = await score_candidates(candidates, selected_cluster)
    
    # Filter low scores
    high_quality_results = [r for r in scored_results if r.score >= 7.0]

    if not high_quality_results:
        console.print("[yellow]No high-quality repositories found (Score >= 7.0).[/yellow]")
        return

    console.print(f"\n[bold green]Found {len(high_quality_results)} top recommendations:[/bold green]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Score")
    table.add_column("Reasoning")
    table.add_column("URL")
    
    for item in high_quality_results:
        table.add_row(
            item.repo.full_name,
            f"{item.score:.1f}",
            item.reasoning,
            f"[link={item.repo.url}]Link[/link]"
        )
    
    console.print(table)
    
    # Placeholder for future stages
    console.print("\n[dim]Actions stage (Star/Ingest) is not yet implemented.[/dim]")

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
