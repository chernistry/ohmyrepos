# GitHub Discovery Agent Architecture

## Overview
The GitHub Discovery Agent is an autonomous module for OhMyRepos that explores GitHub to recommend relevant repositories based on the user's interests. It follows a multi-stage pipeline inspired by CallQuest and Voyant, leveraging LLMs for profiling, filtering, and quality assessment.

## Core Concepts
-   **User Profile**: Derived from existing starred repositories (`repos.json`) and enriched metadata. Represents clusters of interests (e.g., "Machine Learning", "Rust CLIs").
-   **Discovery Pipeline**: A sequential process of finding, filtering, scoring, and verifying repositories.
-   **Observability**: Detailed logging of *why* a repository was chosen or rejected ("Receipts").

## Architecture Components

### 1. Module Structure
The agent will be implemented in `src/agent/discovery.py` and integrated into the CLI via `src/cli.py`.

```
src/
  agent/
    __init__.py
    discovery.py       # Main agent logic and pipeline orchestration
    profile.py         # User profiling and interest clustering
    search.py          # GitHub Search API wrapper (specialized for discovery)
    scoring.py         # LLM-based quality scoring and relevance check
    actions.py         # Actions (Star, Ingest)
  core/
    ...                # Reusing existing core modules (storage, logging, llm)
```

### 2. Pipeline Stages

#### Stage 1: Profile Analysis (`profile.py`)
-   **Input**: `repos.json` (local starred repos).
-   **Process**:
    -   Analyze tags and topics from starred repos.
    -   Use simple clustering (or LLM summarization) to identify top interest categories.
    -   *Output*: List of interest clusters (e.g., `{"topic": "RAG", "languages": ["Python"], "keywords": ["vector", "embedding"]}`).

#### Stage 2: Interactive Selection (CLI)
-   **Process**:
    -   Present identified clusters to the user.
    -   Allow user to select a focus area or define a custom one.
    -   *Output*: Search query parameters (keywords, language, min_stars, date_range).

#### Stage 3: GitHub Search (`search.py`)
-   **Input**: Search parameters.
-   **Process**:
    -   Query GitHub Search API (public repos).
    -   Apply filters: `stars > 100`, `pushed > 2024-01-01` (configurable).
    -   **Dedup**: Check against `repos.json` (already starred) and Qdrant (already seen/rejected).
    -   *Output*: List of candidate repositories (metadata only).

#### Stage 4: Quality Scoring (`scoring.py`)
-   **Input**: Candidate repo metadata + README (fetched via `read_url_content` or GitHub API).
-   **Process**:
    -   **Heuristics**: Check activity (commits, issues), freshness.
    -   **LLM Eval**: Prompt LLM to score relevance (0-10) based on User Profile and README.
    -   *Criteria*: Problem solved, code quality (inferred), documentation quality, relevance to specific user interest.
    -   *Output*: Scored candidates.

#### Stage 5: Verification
-   **Process**:
    -   (Optional) Deep dive: Fetch file structure or `CONTRIBUTING.md`.
    -   Final "sanity check" by LLM or User (interactive mode).
    -   *Output*: Final list of approved repos.

#### Stage 6: Action (`actions.py`)
-   **Process**:
    -   **Star**: Star the repo on GitHub (requires PAT with `public_repo` scope).
    -   **Ingest**: Trigger OhMyRepos ingestion (add to `repos.json`, fetch details, embed, upsert to Qdrant).
    -   *Output*: Updated local state.

### 3. Observability & Receipts
-   Log every decision:
    -   `SEARCH_QUERY`: What was searched.
    -   `CANDIDATE_FOUND`: Repo URL.
    -   `SCORE`: LLM score and *reasoning* (saved to `receipts.jsonl` or similar).
    -   `ACTION`: Starred/Skipped.
-   CLI command `ohmyrepos agent why <repo_url>` to show the reasoning.

## Integration Points
-   **CLI**: Add `agent` subcommand group to `src/cli.py`.
-   **Config**: Add `GITHUB_TOKEN` (with write scope if starring is enabled) and `OPENAI/OPENROUTER_KEY` to `src/config.py`.
-   **Storage**: Use `src/core/storage.py` to check for existing repos in Qdrant.

## Technical Constraints
-   **Rate Limits**: Respect GitHub Search API limits. Use caching.
-   **Cost**: Minimize LLM calls. Use cheap models (e.g., `gpt-4o-mini` or local `llama3`) for scoring. Only use expensive models for complex profiling if needed.
-   **Security**: Handle PAT securely via env vars.

## Roadmap (Tickets)
1.  **Scaffolding**: CLI structure, module files.
2.  **Profile Analysis**: Logic to extract interests from `repos.json`.
3.  **Search & Filter**: GitHub Search integration + Dedup logic.
4.  **Scoring Engine**: LLM prompts and scoring logic.
5.  **Action & Observability**: Starring, Ingestion, and Logging.
