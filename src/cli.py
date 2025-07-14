"""Command-line interface for Oh My Repos.

This module provides a command-line interface for the Oh My Repos application.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

# Исправляем импорты для совместимости
try:
    from src.core.collector import RepoCollector
    from src.core.storage import QdrantStore
    from src.core.summarizer import RepoSummarizer
except ImportError:
    from core.collector import RepoCollector
    from core.storage import QdrantStore
    from core.summarizer import RepoSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("ohmyrepos")
console = Console()

app = typer.Typer(help="Oh My Repos CLI")


@app.command()
def summarize(
    repo_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON file with repository data",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to output file for summary (default: stdout)",
    ),
    incremental_save: bool = typer.Option(
        True,
        "--incremental-save/--no-incremental-save",
        help="Save results incrementally after each repository is processed",
    ),
    concurrency: Optional[int] = typer.Option(
        2,
        "--concurrency",
        "-c",
        help="Number of concurrent summarization tasks (default: 2)",
    ),
):
    """Summarize a GitHub repository from a JSON file.
    
    The JSON file should contain at least 'description' and 'readme' fields.
    """
    try:
        # Load repository data
        repo_data = json.loads(repo_file.read_text(encoding="utf-8"))
        
        # Check if the data is a list and process accordingly
        if isinstance(repo_data, list):
            if not repo_data:
                raise ValueError("Repository data is empty")

            console.print(f"Summarizing [bold]{len(repo_data)}[/bold] repositories")
            if concurrency:
                console.print(f"Using concurrency: [bold]{concurrency}[/bold]")
                
            summaries = asyncio.run(_summarize_repos(
                repo_data, 
                output_file=output_file if incremental_save else None,
                concurrency=concurrency
            ))
            result_data = summaries
        else:
            # Single-repository flow (backward-compatible)
            result_data = asyncio.run(_summarize_repo(repo_data))

        # Output result
        if output_file and not incremental_save:
            output_file.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
            console.print(f"Summary written to [bold]{output_file}[/bold]")
        elif not output_file:
            console.print_json(json.dumps(result_data, indent=2))
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


@app.command()
def generate_summary(
    repo_file: Path = typer.Option(
        "repositories.json",
        "--input",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON file with repositories data (default: repositories.json)",
    ),
    repo_index: int = typer.Option(
        0,
        "--index",
        "-n",
        help="Index of the repository in the JSON array to summarize",
    ),
    repo_name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Name of the repository to summarize (alternative to index)",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to output file for summary (default: stdout)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Show debug information including raw LLM response",
    ),
):
    """Generate a summary for a specific repository from a collection.
    
    This command is useful for debugging the summarization process.
    It loads repositories from a JSON file (default: repositories.json),
    selects one repository by index or name, and generates a summary.
    """
    try:
        # Load repositories data
        console.print(f"Loading repositories from [bold]{repo_file}[/bold]")
        repos_data = json.loads(repo_file.read_text(encoding="utf-8"))
        
        if not repos_data:
            console.print("[bold red]No repositories found in the file[/bold red]")
            sys.exit(1)
        
        # Select repository
        repo_data = None
        if repo_name:
            # Find by name
            for repo in repos_data:
                if repo.get("name") == repo_name:
                    repo_data = repo
                    break
            if not repo_data:
                console.print(f"[bold red]Repository '{repo_name}' not found[/bold red]")
                sys.exit(1)
        else:
            # Get by index
            if repo_index < 0 or repo_index >= len(repos_data):
                console.print(f"[bold red]Invalid repository index: {repo_index}[/bold red]")
                console.print(f"Valid range: 0-{len(repos_data)-1}")
                sys.exit(1)
            repo_data = repos_data[repo_index]
        
        # Show repository info
        console.print(f"Generating summary for repository: [bold]{repo_data.get('name', 'unknown')}[/bold]")
        
        # Enable debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("httpx").setLevel(logging.INFO)
        
        # Run summarization
        console.print("Sending to LLM for summarization...")
        with console.status("[bold green]Generating summary..."):
            summary = asyncio.run(_summarize_repo(repo_data, debug=debug))
        
        # Show debug info
        if debug:
            console.print("\n[bold]Debug Information:[/bold]")
            console.print(f"Repository description length: {len(repo_data.get('description', ''))}")
            console.print(f"Repository README length: {len(repo_data.get('readme', ''))}")
            console.print(f"Generated summary length: {len(summary.get('summary', ''))}")
            console.print(f"Number of tags: {len(summary.get('tags', []))}")
        
        # Output result
        console.print("\n[bold]Generated Summary:[/bold]")
        console.print(summary.get("summary", "No summary generated"))
        
        console.print("\n[bold]Tags:[/bold]")
        console.print(", ".join(summary.get("tags", [])))
        
        # Save to file if requested
        if output_file:
            output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            console.print(f"\nSummary written to [bold]{output_file}[/bold]")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if debug:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@app.command()
def collect(
    output_file: Path = typer.Option(
        "repositories.json",
        "--output",
        "-o",
        help="Path to output file for collected repositories",
    ),
):
    """Collect starred repositories from GitHub.
    
    This command fetches all repositories starred by the configured GitHub user.
    """
    try:
        # Run collection
        repos = asyncio.run(_collect_repos())
        
        # Save to file
        output_file.write_text(json.dumps(repos, indent=2), encoding="utf-8")
        console.print(f"Collected [bold]{len(repos)}[/bold] repositories to [bold]{output_file}[/bold]")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


async def _collect_repos() -> List[Dict[str, Any]]:
    """Collect repositories using the RepoCollector.
    
    Returns:
        List of repository data
    """
    collector = RepoCollector()
    try:
        return await collector.collect_starred_repos()
    finally:
        await collector.close()


async def _summarize_repo(repo_data: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """Summarize a repository using the RepoSummarizer.
    
    Args:
        repo_data: Repository data
        debug: Whether to enable debug mode
        
    Returns:
        Dictionary with summary and tags
    """
    summarizer = RepoSummarizer(debug=debug)
    try:
        result = await summarizer.summarize(repo_data)
        return result
    finally:
        await summarizer.close()


async def _summarize_repos(
    repos: List[Dict[str, Any]], 
    debug: bool = False, 
    output_file: Optional[Path] = None,
    concurrency: int = 2,
) -> List[Dict[str, Any]]:
    """Summarize a list of repositories.

    Args:
        repos: List of repository objects
        debug: Whether to enable verbose logging
        output_file: Optional path to save incremental results
        concurrency: Number of concurrent summarization tasks

    Returns:
        The same list enriched with ``summary`` and ``tags`` fields.
    """
    summarizer = RepoSummarizer(debug=debug, concurrency=concurrency)
    enriched_repos: List[Dict[str, Any]] = []
    skipped_count = 0
    processed_count = 0

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task_id = progress.add_task("Summarizing repositories...", total=len(repos))
            
            # Use batch processing if concurrency is enabled
            if concurrency is not None and concurrency > 1:
                progress.update(task_id, description=f"Summarizing repositories in parallel (concurrency={concurrency})...")
                
                # Process in batch with concurrency and pass output_file for incremental saving
                enriched_repos = await summarizer.summarize_batch(repos, output_file=output_file)
                
                # Count processed and skipped
                for repo in enriched_repos:
                    # Check if the repository has a valid summary and tags
                    has_summary = repo.get("summary") and isinstance(repo.get("summary"), str) and len(repo.get("summary", "")) > 10
                    has_tags = repo.get("tags") and isinstance(repo.get("tags"), list) and len(repo.get("tags", [])) > 0
                    
                    if has_summary and has_tags:
                        if "error" in repo:
                            console.print(f"[bold yellow]⚠[/bold yellow] [bold]{repo.get('name', 'unknown')}[/bold]: {repo.get('error')}")
                        else:
                            processed_count += 1
                            console.print(f"[bold green]✓[/bold green] [bold]{repo.get('name', 'unknown')}[/bold]: {repo.get('summary', '')[:100]}...")
                            console.print(f"   [bold]Tags:[/bold] {', '.join(repo.get('tags', []))}\n")
                    else:
                        skipped_count += 1
                
                # Update progress to completion
                progress.update(task_id, completed=len(repos))
            else:
                # Original sequential processing
                for repo in repos:
                    repo_name = repo.get("name", "unknown")
                    
                    # Skip already-summarized entries
                    has_summary = repo.get("summary") and isinstance(repo.get("summary"), str) and len(repo.get("summary", "")) > 10
                    has_tags = repo.get("tags") and isinstance(repo.get("tags"), list) and len(repo.get("tags", [])) > 0
                    
                    if has_summary and has_tags:
                        enriched_repos.append(repo)
                        skipped_count += 1
                        progress.update(task_id, advance=1, description=f"Skipped {repo_name} (already summarized)")
                        continue

                    progress.update(task_id, description=f"Summarizing {repo_name}...")
                    summary = await summarizer.summarize(repo)
                    repo.update(summary)
                    enriched_repos.append(repo)
                    processed_count += 1
                    
                    # Show summary in console
                    console.print(f"[bold green]✓[/bold green] [bold]{repo_name}[/bold]: {summary.get('summary', '')[:100]}...")
                    console.print(f"   [bold]Tags:[/bold] {', '.join(summary.get('tags', []))}\n")
                    
                    # Save incrementally if requested
                    if output_file:
                        output_file.write_text(json.dumps(enriched_repos, indent=2), encoding="utf-8")
                        
                    progress.update(task_id, advance=1)

        # Print summary statistics
        console.print(f"\n[bold]Summary:[/bold] Processed {processed_count} repositories, skipped {skipped_count} repositories")
        
        return enriched_repos
    finally:
        await summarizer.close()


async def _process_repos(
    input_file: Optional[Path], 
    skip_collection: bool, 
    debug: bool = False,
    output_file: Optional[Path] = None,
    concurrency: Optional[int] = 2,
) -> List[Dict[str, Any]]:
    """Process repositories through the full pipeline.
    
    Args:
        input_file: Optional input file with repository data
        skip_collection: Whether to skip collection
        debug: Whether to enable debug mode
        output_file: Optional path to save incremental results
        concurrency: Number of concurrent summarization tasks
        
    Returns:
        List of enriched repository data
    """
    # Step 1: Collect repositories or load from file
    repos = []
    if input_file and skip_collection:
        console.print(f"Loading repositories from [bold]{input_file}[/bold]")
        repos = json.loads(input_file.read_text(encoding="utf-8"))
    else:
        console.print("Collecting repositories from GitHub")
        collector = RepoCollector()
        try:
            repos = await collector.collect_starred_repos()
        finally:
            await collector.close()
        
        if input_file and repos:
            # Merge with existing data if available
            try:
                existing = json.loads(input_file.read_text(encoding="utf-8"))
                # Create a map of existing repos by name
                existing_map = {repo.get("name", ""): repo for repo in existing}
                # Update with new data
                for repo in repos:
                    name = repo.get("name", "")
                    if name in existing_map:
                        existing_map[name].update(repo)
                repos = list(existing_map.values())
            except (json.JSONDecodeError, FileNotFoundError):
                pass
    
    if not repos:
        console.print("[bold red]No repositories found[/bold red]")
        return []
    
    # Step 2: Summarize repositories
    console.print(f"Summarizing [bold]{len(repos)}[/bold] repositories")
    if concurrency:
        console.print(f"Using concurrency: [bold]{concurrency}[/bold]")
        
    summarizer = RepoSummarizer(debug=debug, concurrency=concurrency)
    
    try:
        # Use batch processing if concurrency is enabled
        if concurrency is not None and concurrency > 1:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            ) as progress:
                summarize_task = progress.add_task(
                    f"Summarizing repositories in parallel (concurrency={concurrency})...", 
                    total=len(repos)
                )
                
                # Process in batch with concurrency and pass output_file for incremental saving
                enriched_repos = await summarizer.summarize_batch(repos, output_file=output_file)
                
                # Show summaries
                processed_count = 0
                skipped_count = 0
                
                for repo in enriched_repos:
                    # Check if the repository has a valid summary and tags
                    has_summary = repo.get("summary") and isinstance(repo.get("summary"), str) and len(repo.get("summary", "")) > 10
                    has_tags = repo.get("tags") and isinstance(repo.get("tags"), list) and len(repo.get("tags", [])) > 0
                    
                    if has_summary and has_tags:
                        repo_name = repo.get('name', 'unknown')
                        if "error" in repo:
                            console.print(f"[bold yellow]⚠[/bold yellow] [bold]{repo_name}[/bold]: {repo.get('error')}")
                        else:
                            processed_count += 1
                            console.print(f"[bold green]✓[/bold green] [bold]{repo_name}[/bold]: {repo.get('summary', '')[:100]}...")
                            console.print(f"   [bold]Tags:[/bold] {', '.join(repo.get('tags', []))}\n")
                    else:
                        skipped_count += 1
                
                # Update progress to completion
                progress.update(summarize_task, completed=len(repos))
        else:
            # Original sequential processing
            enriched_repos = []
            skipped_count = 0
            processed_count = 0
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            ) as progress:
                summarize_task = progress.add_task("Summarizing repositories...", total=len(repos))
                
                for repo in repos:
                    repo_name = repo.get('name', 'unknown')
                    
                    # Skip if already summarized
                    has_summary = repo.get("summary") and isinstance(repo.get("summary"), str) and len(repo.get("summary", "")) > 10
                    has_tags = repo.get("tags") and isinstance(repo.get("tags"), list) and len(repo.get("tags", [])) > 0
                    
                    if has_summary and has_tags:
                        enriched_repos.append(repo)
                        skipped_count += 1
                        progress.update(summarize_task, advance=1, description=f"Skipped {repo_name} (already summarized)")
                        continue
                    
                    # Update task description
                    progress.update(
                        summarize_task, 
                        description=f"Summarizing {repo_name}..."
                    )
                    
                    # Summarize
                    summary = await summarizer.summarize(repo)
                    repo.update(summary)
                    enriched_repos.append(repo)
                    processed_count += 1
                    
                    # Show summary in console
                    console.print(f"[bold green]✓[/bold green] [bold]{repo_name}[/bold]: {summary.get('summary', '')[:100]}...")
                    console.print(f"   [bold]Tags:[/bold] {', '.join(summary.get('tags', []))}\n")
                    
                    # Save incrementally if requested
                    if output_file:
                        output_file.write_text(json.dumps(enriched_repos, indent=2), encoding="utf-8")
                    
                    # Update progress
                    progress.update(summarize_task, advance=1)
            
            # Print summary statistics
            console.print(f"\n[bold]Summary:[/bold] Processed {processed_count} repositories, skipped {skipped_count} repositories")
    finally:
        await summarizer.close()
    
    # Step 3: Store in vector database
    console.print("Storing repositories in vector database")
    store = QdrantStore()
    
    try:
        await store.setup_collection()
        # Обновляем enriched_repos с информацией о статусе эмбеддингов
        enriched_repos = await store.store_repositories(enriched_repos)
        
        # Статистика по эмбеддингам
        new_embeddings = sum(1 for repo in enriched_repos if repo.get("has_embedding", False))
        skipped_embeddings = len(enriched_repos) - new_embeddings
        
        console.print(f"[bold]Embedding status:[/bold] {new_embeddings} repositories with embeddings, {skipped_embeddings} skipped (already had embeddings)")
        
        # Сохраняем обновленный JSON с информацией о статусе эмбеддингов
        if output_file:
            output_file.write_text(json.dumps(enriched_repos, indent=2), encoding="utf-8")
            console.print(f"Updated repository data with embedding status saved to [bold]{output_file}[/bold]")
    finally:
        await store.close()
    
    console.print("[bold green]Repository processing complete![/bold green]")
    return enriched_repos


async def _embed_only_repos(
    input_file: Path,
    output_file: Optional[Path] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Generate embeddings for repositories that already have summaries.
    
    Args:
        input_file: Input file with repository data (must include summaries)
        output_file: Optional path to save incremental results
        debug: Whether to enable debug mode
        
    Returns:
        List of repository data with embedding status
    """
    # Load repositories from file
    repos = json.loads(input_file.read_text(encoding="utf-8"))
    
    if not repos:
        console.print("[bold red]No repositories found in the input file[/bold red]")
        return []
    
    # Filter repositories that have summaries but no embeddings
    valid_repos = []
    skipped_no_summary = 0
    skipped_has_embedding = 0
    
    for repo in repos:
        # Check if repository has valid summary and tags
        has_summary = repo.get("summary") and isinstance(repo.get("summary"), str) and len(repo.get("summary", "")) > 10
        has_tags = repo.get("tags") and isinstance(repo.get("tags"), list) and len(repo.get("tags", [])) > 0
        has_embedding = repo.get("has_embedding", False)
        
        if not has_summary or not has_tags:
            skipped_no_summary += 1
            continue
            
        if has_embedding:
            skipped_has_embedding += 1
            continue
            
        valid_repos.append(repo)
    
    if not valid_repos:
        console.print("[bold yellow]No repositories need embedding generation[/bold yellow]")
        console.print(f"Skipped: {skipped_no_summary} without summaries, {skipped_has_embedding} already have embeddings")
        return repos
    
    console.print(f"Found [bold]{len(valid_repos)}[/bold] repositories that need embeddings")
    console.print(f"Skipped: {skipped_no_summary} without summaries, {skipped_has_embedding} already have embeddings")
    
    # Store in vector database
    console.print("Storing repositories in vector database")
    store = QdrantStore()
    
    try:
        await store.setup_collection()
        # Process all repositories to update embedding status
        updated_repos = await store.store_repositories(repos)
        
        # Статистика по эмбеддингам
        new_embeddings = sum(1 for repo in updated_repos if repo.get("has_embedding", False) and repo.get("name", "") in [r.get("name", "") for r in valid_repos])
        
        console.print(f"[bold]Embedding status:[/bold] {new_embeddings} new repositories with embeddings")
        
        # Сохраняем обновленный JSON с информацией о статусе эмбеддингов
        if output_file:
            output_file.write_text(json.dumps(updated_repos, indent=2), encoding="utf-8")
            console.print(f"Updated repository data with embedding status saved to [bold]{output_file}[/bold]")
        
        return updated_repos
    
    finally:
        await store.close()


@app.command()
def embed_only(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON file with repository data (must include summaries)",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to output file for updated repositories (default: overwrite input file)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Show debug information",
    ),
):
    """Generate embeddings only for repositories that already have summaries.
    
    This command skips the collection and summarization steps, and only generates
    embeddings for repositories that already have valid summaries and tags.
    """
    try:
        # Enable debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("httpx").setLevel(logging.INFO)
        
        # If no output file specified, use the input file
        actual_output = output_file or input_file
        
        # Run embedding generation
        updated_repos = asyncio.run(_embed_only_repos(
            input_file=input_file,
            output_file=actual_output,
            debug=debug
        ))
        
        console.print("[bold green]Embedding generation complete![/bold green]")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if debug:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@app.command()
def embed(
    input_file: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON file with repository data (optional, will collect if not provided)",
    ),
    skip_collection: bool = typer.Option(
        False,
        "--skip-collection",
        "-s",
        help="Skip collection and use only the provided input file",
    ),
    output_file: Optional[Path] = typer.Option(
        "enriched_repos.json",
        "--output",
        "-o",
        help="Path to output file for enriched repositories",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Show debug information including raw LLM responses",
    ),
    incremental_save: bool = typer.Option(
        True,
        "--incremental-save/--no-incremental-save",
        help="Save results incrementally after each repository is processed",
    ),
    concurrency: Optional[int] = typer.Option(
        2,
        "--concurrency",
        "-c",
        help="Number of concurrent summarization tasks (default: 2)",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--reprocess-all",
        help="Skip repositories that already have embeddings in the vector database",
    ),
):
    """Process repositories: collect, summarize, and store in vector database.
    
    This command runs the full pipeline: collection, summarization, and storage.
    """
    try:
        # Enable debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("httpx").setLevel(logging.INFO)
        
        # Run the full pipeline
        enriched_repos = asyncio.run(_process_repos(
            input_file, 
            skip_collection, 
            debug=debug,
            output_file=output_file if incremental_save else None,
            concurrency=concurrency
        ))
        
        # Save enriched data if not incremental
        if output_file and enriched_repos and not incremental_save:
            output_file.write_text(json.dumps(enriched_repos, indent=2), encoding="utf-8")
            console.print(f"Processed [bold]{len(enriched_repos)}[/bold] repositories to [bold]{output_file}[/bold]")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if debug:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@app.command()
def serve(
    host: str = typer.Option(
        "localhost",
        "--host",
        help="Host to serve the UI on",
    ),
    port: int = typer.Option(
        8501,
        "--port",
        "-p",
        help="Port to serve the UI on",
    ),
):
    """Launch the Streamlit UI for searching repositories.
    
    This command starts a Streamlit server that provides a web interface
    for searching through your GitHub starred repositories.
    """
    import os
    import subprocess
    
    # Get the path to the app.py file
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        console.print(f"[bold red]Error: UI file not found at {app_path}[/bold red]")
        sys.exit(1)
    
    console.print(f"Starting Streamlit UI on [bold]{host}:{port}[/bold]")
    
    # Set environment variables for Streamlit
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_SERVER_ADDRESS"] = host
    
    # Run Streamlit
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            env=env,
            check=True,
        )
    except KeyboardInterrupt:
        console.print("\n[bold green]UI server stopped[/bold green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error running Streamlit: {e}[/bold red]")
        sys.exit(1)


@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Search query",
    ),
    limit: int = typer.Option(
        25,
        "--limit",
        "-l",
        help="Maximum number of results to return",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to output file for search results (default: stdout)",
    ),
    filter_tags: Optional[List[str]] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Filter results by tag (can be specified multiple times)",
    ),
):
    """Search for repositories using hybrid retrieval.
    
    This command performs a hybrid search (vector + BM25) on the repository collection
    and returns the most relevant results.
    """
    from src.core.retriever import HybridRetriever
    from src.core.reranker import JinaReranker
    
    try:
        # Initialize retriever and reranker
        console.print(f"Searching for: [bold]{query}[/bold]")
        
        with console.status("[bold green]Initializing retriever..."):
            retriever = HybridRetriever()
            asyncio.run(retriever.initialize())
        
        # Initialize reranker
        reranker = JinaReranker()
        
        # Perform search
        with console.status("[bold green]Searching repositories..."):
            # Search with retriever
            results = asyncio.run(retriever.search(query, limit=limit*2, filter_tags=filter_tags))
            console.print(f"Found [bold]{len(results)}[/bold] initial results")
            
            # Rerank if we have more than 1 result
            if len(results) > 1:
                with console.status("[bold green]Reranking results..."):
                    results = asyncio.run(reranker.rerank(query, results, top_k=limit))
        
        # Output results
        if output_file:
            output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
            console.print(f"Search results written to [bold]{output_file}[/bold]")
        else:
            # Format for console display
            console.print(f"\n[bold]Top {len(results)} results:[/bold]")
            for i, result in enumerate(results):
                console.print(f"\n[bold cyan]{i+1}.[/bold cyan] [bold link={result['repo_url']}]{result['repo_name']}[/bold link]")
                console.print(f"Score: {result['score']:.3f}")
                
                if "summary" in result and result["summary"]:
                    console.print(result["summary"])
                
                if "language" in result and result["language"]:
                    console.print(f"Language: [bold]{result['language']}[/bold]")
                
                if "tags" in result and result["tags"]:
                    console.print(f"Tags: {', '.join(result['tags'])}")
        
        # Clean up
        asyncio.run(retriever.close())
        asyncio.run(reranker.close())
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    app()
