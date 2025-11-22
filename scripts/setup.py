#!/usr/bin/env python3
"""Interactive setup helper for Oh My Repos.

Goals:
- Collect required secrets and provider choices.
- Default to Ollama embeddings (auto-select after 10s).
- Ensure the Ollama embedding model is available if chosen.
- Optionally run collection + embedding so the stack is ready to search.
"""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

# ANSI colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def info(msg: str) -> None:
    print(f"{BLUE}ℹ{RESET} {msg}")

def success(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")

def warning(msg: str) -> None:
    print(f"{YELLOW}⚠{RESET} {msg}")

def error(msg: str) -> None:
    print(f"{RED}✗{RESET} {msg}")

def header(msg: str) -> None:
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{msg.center(60)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")

REQUIRED_KEYS = [
    "GITHUB_USERNAME",
    "GITHUB_TOKEN",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "CHAT_LLM_PROVIDER",
    "CHAT_LLM_BASE_URL",
    "CHAT_LLM_MODEL",
    "CHAT_LLM_API_KEY",
]

EMBEDDING_KEYS = [
    "EMBEDDINGS_SERVICE",
    "EMBEDDING_MODEL",
    "EMBEDDING_MODEL_URL",
    "EMBEDDING_MODEL_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBEDDING_MODEL",
]

ORDER = REQUIRED_KEYS + EMBEDDING_KEYS + [
    "RERANKER_API_KEY",
    "ENVIRONMENT",
    "DEBUG",
    "LOG_LEVEL",
    "QDRANT_COLLECTION_NAME",
]

env: Dict[str, str] = {}


def read_env(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        striped = line.strip()
        if not striped or striped.startswith("#") or "=" not in striped:
            continue
        key, val = striped.split("=", 1)
        data[key.strip()] = val.strip()
    return data


def write_env(env: Dict[str, str]) -> None:
    lines = []
    for key in ORDER:
        if key in env:
            lines.append(f"{key}={env[key]}")
    for key, val in sorted(env.items()):
        if key in ORDER:
            continue
        lines.append(f"{key}={val}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    success(f"Wrote environment to {ENV_PATH}")


def input_with_timeout(prompt: str, default: str, timeout: int = 10) -> str:
    """POSIX-only timeout input helper (falls back to default on timeout)."""
    print(f"{prompt} [{default}] (auto in {timeout}s): ", end="", flush=True)
    try:
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            value = sys.stdin.readline().strip()
            return value or default
    except Exception:
        # Fallback: no timeout support (use default)
        return default
    print()  # newline after timeout
    return default


def prompt(key: str, message: str, default: Optional[str] = None, secret: bool = False, timeout: Optional[int] = None) -> str:
    existing = os.getenv(key) or env.get(key)
    if existing:
        info(f"{key} already set, keeping existing value.")
        return existing

    if timeout is not None:
        return input_with_timeout(message, default or "", timeout=timeout)

    suffix = f" [{default}]" if default else ""
    value = input(f"{message}{suffix}: ").strip()
    return value or (default or "")


def ensure_ollama_model(model: str) -> None:
    if not shutil.which("ollama"):
        print("[warn] Ollama CLI not found. Install from https://ollama.com/download and rerun to pull models.")
        return

    try:
        listed = subprocess.check_output(["ollama", "list"], text=True)
        if model in listed:
            success(f"Ollama model '{model}' already present.")
            return
    except Exception:
        pass

    info(f"Pulling Ollama model '{model}' (one-time)...")
    subprocess.run(["ollama", "pull", model], check=False)


def run_cmd(cmd: list[str], cwd: Path = ROOT, label: str = "") -> bool:
    label_prefix = f"[{label}] " if label else ""
    print(f"{label_prefix}running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        warning(f"command failed with exit code {result.returncode}")
        return False
    return True


def yes_no(message: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    import sys
    choice = sys.stdin.readline().strip().lower()
    if not choice:
        print(f"{message} [{suffix}]: ", end='', flush=True)
        choice = sys.stdin.readline().strip().lower()
    if not choice:
        return default
    return choice in ("y", "yes")


def configure_embeddings(env: Dict[str, str]) -> None:
    choice = prompt(
        "EMBEDDINGS_SERVICE",
        "Embedding provider (ollama/jina)",
        default="ollama",
        timeout=10,
    ).lower()

    if choice not in ("ollama", "jina"):
        print("[warn] Invalid choice, defaulting to ollama.")
        choice = "ollama"

    if choice == "ollama":
        env["EMBEDDINGS_SERVICE"] = "ollama"
        env["EMBEDDING_MODEL"] = "embeddinggemma:latest"
        env["OLLAMA_EMBEDDING_MODEL"] = "embeddinggemma:latest"
        env["OLLAMA_BASE_URL"] = env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/api/embeddings")
        env["EMBEDDING_MODEL_URL"] = env.get("EMBEDDING_MODEL_URL", "http://127.0.0.1:11434/api/embeddings")
        ensure_ollama_model("embeddinggemma:latest")
    else:
        env["EMBEDDINGS_SERVICE"] = "jina"
        env["EMBEDDING_MODEL"] = env.get("EMBEDDING_MODEL", "jina-embeddings-v3")
        env["EMBEDDING_MODEL_URL"] = env.get("EMBEDDING_MODEL_URL", "https://api.jina.ai/v1/embeddings")
        env["EMBEDDING_MODEL_API_KEY"] = prompt(
            "EMBEDDING_MODEL_API_KEY",
            "Jina embeddings API key",
            default=env.get("EMBEDDING_MODEL_API_KEY", ""),
        )
        env["RERANKER_API_KEY"] = env.get("RERANKER_API_KEY", env["EMBEDDING_MODEL_API_KEY"])


def configure_required(env: Dict[str, str]) -> None:
    env["GITHUB_USERNAME"] = prompt("GITHUB_USERNAME", "GitHub username", env.get("GITHUB_USERNAME", ""))
    env["GITHUB_TOKEN"] = prompt("GITHUB_TOKEN", "GitHub token (repo read scope)", env.get("GITHUB_TOKEN", ""))

    env["QDRANT_URL"] = prompt("QDRANT_URL", "Qdrant URL", env.get("QDRANT_URL", "http://localhost:6333"))
    env["QDRANT_API_KEY"] = prompt("QDRANT_API_KEY", "Qdrant API key (blank for local)", env.get("QDRANT_API_KEY", ""))

    env["CHAT_LLM_PROVIDER"] = prompt("CHAT_LLM_PROVIDER", "Chat LLM provider", env.get("CHAT_LLM_PROVIDER", "openai"))
    env["CHAT_LLM_BASE_URL"] = prompt("CHAT_LLM_BASE_URL", "Chat LLM base URL", env.get("CHAT_LLM_BASE_URL", "https://openrouter.ai/api/v1"))
    env["CHAT_LLM_MODEL"] = prompt("CHAT_LLM_MODEL", "Chat LLM model", env.get("CHAT_LLM_MODEL", "deepseek/deepseek-r1-0528:free"))
    env["CHAT_LLM_API_KEY"] = prompt("CHAT_LLM_API_KEY", "Chat LLM API key", env.get("CHAT_LLM_API_KEY", ""))


def run_ingestion() -> None:
    if not yes_no("Fetch starred repositories now?", default=True):
        return
    if not run_cmd([sys.executable, "ohmyrepos.py", "collect", "--output", "repos.json", "--incremental"], label="collect"):
        return

    if yes_no("Generate summaries + embeddings and push to Qdrant now?", default=True):
        run_cmd(
            [
                sys.executable,
                "ohmyrepos.py",
                "embed",
                "--input",
                "repos.json",
                "--skip-collection",
                "--concurrency",
                "4",
                "--output",
                "enriched_repos.json",
            ],
            label="embed",
        )


if __name__ == "__main__":
    env = read_env(ENV_PATH)
    info(f"Using repository root: {ROOT}")
    configure_embeddings(env)
    configure_required(env)
    write_env(env)
    run_ingestion()

    print("\nNext steps:")
    print("1) Start dev stack: ./run.sh dev")
    print("2) Or start only backend/frontend: ./run.sh b start | ./run.sh f start")
    print("3) To expose MCP tools (Claude/Cursor): python mcp_server.py")
