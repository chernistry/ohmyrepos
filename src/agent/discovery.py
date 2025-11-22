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
    skip_profile: bool = False,
    auto_mode: bool = False,
    limit: int = 10
):
    """Run the discovery pipeline.

    Args:
        repos_path: Path to the local repositories JSON file.
        max_results: Maximum number of repositories to recommend.
        category: Optional specific category to focus on.
        skip_profile: If True, skip profile analysis and go straight to search.
        auto_mode: If True, automatically star high-quality repositories.
        limit: Maximum number of repositories to star in auto mode.
    """
    asyncio.run(_discover_async(repos_path, max_results, category, skip_profile, auto_mode, limit))

async def _discover_async(
    repos_path: Path,
    max_results: int = 20,
    category: Optional[str] = None,
    skip_profile: bool = False,
    auto_mode: bool = False,
    limit: int = 10
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
    
    # Track state for continuous loop
    starred_count = 0
    starred_repos = set()
    current_intent = intent
    previous_intents = [intent]
    
    if auto_mode:
        console.print(f"\n[bold magenta]Running in Autonomous Mode (Limit: {limit})[/bold magenta]")
        
        while starred_count < limit:
            # If this is not the first iteration, we need to search again with evolved intent
            if len(previous_intents) > 1:
                console.print(f"\n[bold blue]Evolving Search...[/bold blue]")
                console.print(f"New Intent: '{current_intent}'")
                
                # Update excluded repos with what we found so far
                excluded.update(starred_repos)
                
                candidates = await smart_search(
                    intent=current_intent,
                    max_results=max_results,
                    excluded_repos=excluded
                )
                
                if not candidates:
                    console.print("[yellow]No new candidates found with this intent. Trying to evolve again...[/yellow]")
                    # Force evolution even if no candidates found
                else:
                    console.print(f"\n[bold]Scoring {len(candidates)} new candidates...[/bold]")
                    scored_results = await score_candidates(candidates, selected_cluster)
                    high_quality_results = [r for r in scored_results if r.score >= 7.0]
            
            # Adaptive Thresholding Loop for current batch
            thresholds = [8.5, 7.5, 7.0]
            batch_starred = 0
            
            for threshold in thresholds:
                if starred_count >= limit:
                    break
                
                # If we already starred something in this batch (at higher threshold), 
                # we might not need to lower threshold if we are happy with quality.
                # But to maximize recall, we can continue.
                # Let's stick to: if we found 0 at 8.5, try 7.5.
                if batch_starred > 0:
                    break
                    
                console.print(f"[cyan]Checking candidates with score >= {threshold}...[/cyan]")
                
                for item in high_quality_results:
                    if starred_count >= limit:
                        break
                    
                    if item.repo.full_name in starred_repos:
                        continue
                        
                    if item.score >= threshold:
                        console.print(f"Auto-starring [bold]{item.repo.full_name}[/bold] (Score: {item.score:.1f})...")
                        if await action_manager.star_repo(item.repo.full_name):
                            console.print(f"[green]Successfully starred {item.repo.full_name}![/green]")
                            starred_count += 1
                            batch_starred += 1
                            starred_repos.add(item.repo.full_name)
                            excluded.add(item.repo.full_name) # Add to excluded immediately
                        else:
                            console.print(f"[red]Failed to star {item.repo.full_name}.[/red]")
            
            if starred_count >= limit:
                break
                
            # Evolve Intent for next iteration
            from src.agent.search import evolve_intent
            console.print("[dim]Thinking about next search step...[/dim]")
            
            # Collect names of found repos to avoid
            found_names = [r.repo.full_name for r in high_quality_results]
            
            new_intent = await evolve_intent(intent, previous_intents, found_names)
            
            if new_intent == current_intent or new_intent in previous_intents:
                console.print("[yellow]Could not generate distinct new intent. Stopping.[/yellow]")
                break
                
            current_intent = new_intent
            previous_intents.append(current_intent)
            
            # Small pause to be nice to APIs
            await asyncio.sleep(2)

        console.print(f"\n[bold]Autonomous session complete. Starred {starred_count} repositories.[/bold]")
        
    else:
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
