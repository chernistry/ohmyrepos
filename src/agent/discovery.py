import asyncio
import logging
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

from src.agent.profile import analyze_profile, InterestCluster
from src.agent.search import smart_search, get_local_repos
from src.agent.scoring import score_candidates
from src.core.storage import QdrantStore

logger = logging.getLogger("ohmyrepos.agent.discovery")
console = Console()

def discover(
    repos_path: Path,
    max_results: int = 20,
    category: Optional[str] = None,
    skip_profile: bool = False
):
    """Run the discovery pipeline.

    Args:
        repos_path: Path to the local repositories JSON file.
        max_results: Maximum number of repositories to recommend.
        category: Optional specific category to focus on.
        skip_profile: If True, skip profile analysis and go straight to search.
    """
    asyncio.run(_discover_async(repos_path, max_results, category, skip_profile))

async def _discover_async(
    repos_path: Path,
    max_results: int = 20,
    category: Optional[str] = None,
    skip_profile: bool = False
):
    console.print("[bold blue]GitHub Discovery Agent[/bold blue]")
    
    clusters = []
    if not skip_profile:
        console.print(f"Analyzing profile from: {repos_path}")

        # Stage 1: Profile Analysis
        clusters = analyze_profile(repos_path)
        
        if not clusters:
            console.print("[yellow]No interest clusters found. Try starring more repos![/yellow]")
            # Don't return here, allow falling through to custom query if desired, 
            # or maybe we should return if not skipping? 
            # The original code returned. Let's keep it consistent but allow skip.
            if not category: # If no category and no clusters, we can't do much unless we force custom query
                 pass 
    
        console.print("\n[bold]Identified Interest Clusters:[/bold]")
        _print_clusters(clusters)
    else:
        console.print("[dim]Skipping profile analysis...[/dim]")

    # Stage 2: Selection
    intent = category
    selected_cluster = None
    
    if not intent:
        # Interactive selection
        choices = [c.name for c in clusters] + ["Custom Query"]
        
        console.print("\n[bold]Select an interest to explore:[/bold]")
        for idx, choice in enumerate(choices, 1):
            console.print(f"{idx}. {choice}")
            
        choice_idx = int(Prompt.ask("Enter number", default="1")) - 1
        
        if 0 <= choice_idx < len(clusters):
            selected_cluster = clusters[choice_idx]
            # Construct intent from cluster
            intent = f"{selected_cluster.name} related to {', '.join(selected_cluster.keywords)}"
            console.print(f"[green]Selected cluster: {selected_cluster.name}[/green]")
        else:
            intent = Prompt.ask("Enter your search intent (e.g., 'Find me RAG agents')")
            # Create a dummy cluster for custom query
            selected_cluster = InterestCluster(name="Custom", keywords=intent.split(), languages=[], score=1.0)

    console.print(f"\n[bold]Searching GitHub for:[/bold] '{intent}'")

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

    candidates = await smart_search(
        intent=intent,
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

    # Interactive Action Loop
    from src.agent.actions import ActionManager
    action_manager = ActionManager()

    while True:
        console.print(f"\n[bold green]Found {len(high_quality_results)} top recommendations:[/bold green]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim")
        table.add_column("Name")
        table.add_column("Score")
        table.add_column("Reasoning")
        table.add_column("URL")
        
        for idx, item in enumerate(high_quality_results, 1):
            table.add_row(
                str(idx),
                item.repo.full_name,
                f"{item.score:.1f}",
                item.reasoning,
                f"[link={item.repo.url}]Link[/link]"
            )
        
        console.print(table)
        
        console.print("\n[bold]Actions:[/bold]")
        console.print("[bold green]S[/bold green] <number>: Star repository (e.g., 'S 1')")
        console.print("[bold blue]I[/bold blue] <number>: Ingest repository (e.g., 'I 1')")
        console.print("[bold yellow]C[/bold yellow]: Continue / Exit")
        
        choice = Prompt.ask("Enter action").strip().upper()
        
        if choice == "C":
            break
            
        try:
            action, idx_str = choice.split(maxsplit=1)
            idx = int(idx_str) - 1
            
            if 0 <= idx < len(high_quality_results):
                repo = high_quality_results[idx].repo
                
                if action == "S":
                    console.print(f"Starring {repo.full_name}...")
                    if await action_manager.star_repo(repo.full_name):
                        console.print(f"[green]Successfully starred {repo.full_name}![/green]")
                    else:
                        console.print(f"[red]Failed to star {repo.full_name}. Check logs.[/red]")
                        
                elif action == "I":
                    console.print(f"Ingesting {repo.full_name}...")
                    if await action_manager.ingest_repo(repo.url):
                        console.print(f"[green]Successfully ingested {repo.full_name}![/green]")
                    else:
                        console.print(f"[red]Failed to ingest {repo.full_name}. Check logs.[/red]")
                else:
                    console.print("[red]Invalid action code.[/red]")
            else:
                console.print("[red]Invalid repository number.[/red]")
                
        except ValueError:
            console.print("[red]Invalid format. Use 'S 1' or 'I 1'.[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

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
