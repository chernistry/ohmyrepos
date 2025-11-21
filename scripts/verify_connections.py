#!/usr/bin/env python3
"""Verify connectivity to all external services."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from rich.console import Console
from rich.table import Table

from src.config import settings

console = Console()


async def check_github() -> Tuple[bool, str]:
    """Check GitHub API connectivity."""
    if not settings.github:
        return False, "GitHub config not found"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.github.api_url}/user",
                headers={
                    "Authorization": f"token {settings.github.token.get_secret_value()}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            if response.status_code == 200:
                data = response.json()
                return True, f"Connected as {data.get('login', 'unknown')}"
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)


async def check_qdrant() -> Tuple[bool, str]:
    """Check Qdrant connectivity."""
    if not settings.qdrant:
        return False, "Qdrant config not found"

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=str(settings.qdrant.url),
            api_key=settings.qdrant.api_key.get_secret_value()
            if settings.qdrant.api_key
            else None,
            timeout=10,
        )
        collections = client.get_collections()
        return True, f"Connected, {len(collections.collections)} collections"
    except Exception as e:
        return False, str(e)


async def check_openrouter() -> Tuple[bool, str]:
    """Check OpenRouter/LLM API connectivity."""
    if not settings.llm or not settings.llm.api_key:
        return False, "LLM config or API key not found"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.llm.base_url}/models",
                headers={
                    "Authorization": f"Bearer {settings.llm.api_key.get_secret_value()}",
                },
            )
            if response.status_code == 200:
                return True, f"Connected, model: {settings.llm.model}"
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)


async def check_jina_embeddings() -> Tuple[bool, str]:
    """Check Jina embeddings API connectivity."""
    if not settings.embedding:
        return False, "Embedding config not found"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                str(settings.embedding.base_url),
                headers={
                    "Authorization": f"Bearer {settings.embedding.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json={"input": ["test"], "model": settings.embedding.model},
            )
            if response.status_code == 200:
                return True, f"Connected, model: {settings.embedding.model}"
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)


async def check_ollama() -> Tuple[bool, str]:
    """Check Ollama connectivity (if configured)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                if settings.ollama.model in models:
                    return True, f"Connected, model {settings.ollama.model} available"
                return (
                    True,
                    f"Connected, but {settings.ollama.model} not found. Available: {', '.join(models[:3])}",
                )
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Not running or unreachable: {e}"


async def main():
    """Run all connectivity checks."""
    console.print("\n[bold cyan]🔍 Verifying Service Connections[/bold cyan]\n")

    checks: List[Tuple[str, asyncio.Task]] = [
        ("GitHub API", asyncio.create_task(check_github())),
        ("Qdrant Vector DB", asyncio.create_task(check_qdrant())),
        ("OpenRouter/LLM", asyncio.create_task(check_openrouter())),
        ("Jina Embeddings", asyncio.create_task(check_jina_embeddings())),
        ("Ollama (optional)", asyncio.create_task(check_ollama())),
    ]

    results: Dict[str, Tuple[bool, str]] = {}
    for name, task in checks:
        try:
            success, message = await task
            results[name] = (success, message)
        except Exception as e:
            results[name] = (False, f"Check failed: {e}")

    # Display results
    table = Table(title="Service Connectivity Status")
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Details")

    all_critical_ok = True
    for name, (success, message) in results.items():
        status = "[green]✓ OK[/green]" if success else "[red]✗ FAIL[/red]"
        table.add_row(name, status, message)

        # Ollama is optional
        if not success and name != "Ollama (optional)":
            all_critical_ok = False

    console.print(table)

    if all_critical_ok:
        console.print("\n[bold green]✓ All critical services are accessible![/bold green]\n")
        return 0
    else:
        console.print(
            "\n[bold red]✗ Some critical services are not accessible. Check configuration.[/bold red]\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
