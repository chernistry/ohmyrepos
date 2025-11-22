"""Main discovery module for GitHub Discovery Agent.

This module orchestrates the discovery pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from src.agent.profile import analyze_profile

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
    console.print("[bold blue]GitHub Discovery Agent[/bold blue]")
    console.print(f"Analyzing profile from: {repos_path}")

    # Stage 1: Profile Analysis
    clusters = analyze_profile(repos_path)
    
    if not clusters:
        console.print("[yellow]No interest clusters found. Try starring more repos![/yellow]")
        return

    console.print("\n[bold]Identified Interest Clusters:[/bold]")
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

    # Placeholder for future stages
    console.print("\n[dim]Search and Scoring stages are not yet implemented.[/dim]")
