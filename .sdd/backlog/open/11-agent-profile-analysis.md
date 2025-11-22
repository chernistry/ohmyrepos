# Ticket 11: Agent Profile Analysis

**Goal**: Implement Stage 1 of the pipeline: analyzing the user's existing starred repositories to build an interest profile.

## Requirements
-   Read `repos.json` (source of truth for starred repos).
-   Extract tags, languages, and topics.
-   Implement a simple clustering mechanism (or use LLM summarization if cheap) to group interests.
-   Output a list of "Interest Clusters" (e.g., "Python/AI", "Rust/CLI").

## Implementation Details
-   **Module**: `src/agent/profile.py`
-   **Function**: `analyze_profile(repos_path: str) -> List[InterestCluster]`
-   **Logic**:
    -   Load JSON.
    -   Counter for languages and topics.
    -   (Optional) Call LLM to summarize top 50 repos into 3-5 personas/clusters.

## Verification
-   Unit test `analyze_profile` with a mock `repos.json`.
-   Run CLI `ohmyrepos agent discover` and verify it prints detected interest clusters.
