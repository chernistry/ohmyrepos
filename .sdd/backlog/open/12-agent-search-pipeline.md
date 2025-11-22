# Ticket 12: Agent Search Pipeline

**Goal**: Implement Stage 2 & 3: Interactive selection and GitHub Search.

## Requirements
-   **Interactive Selection**: Ask user to pick a cluster from Ticket 11 or enter a custom query.
-   **GitHub Search**: Query GitHub API for repositories matching the selection.
-   **Filters**: Apply default filters (stars > 100, pushed > 1 year ago).
-   **Dedup**: Exclude repos already in `repos.json` or Qdrant.

## Implementation Details
-   **Module**: `src/agent/search.py`
-   **Function**: `search_github(query: str, filters: dict) -> List[RepoMetadata]`
-   **Integration**: Connect `profile.py` output to `search.py` input in `discovery.py`.

## Verification
-   Mock GitHub API response.
-   Test deduplication logic (ensure existing repos are filtered out).
-   Manual test: Run CLI, select a category, verify it fetches *new* repos.
