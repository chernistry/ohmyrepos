# Ticket 14: Agent Actions & Observability

**Goal**: Implement Stage 6 (Action) and Stage 3 (Receipts/Observability).

## Requirements
-   **Actions**:
    -   `star_repo(repo_url)`: Star on GitHub (if token allows).
    -   `ingest_repo(repo_url)`: Call core ingestion logic.
-   **Observability**:
    -   Log all decisions to `receipts.jsonl`.
    -   Implement `ohmyrepos agent why <repo_url>` to read from receipts.

## Implementation Details
-   **Module**: `src/agent/actions.py`
-   **Logging**: Use `src/core/logging.py` or dedicated receipts logger.

## Verification
-   Test `star_repo` with a test account or mock.
-   Verify `receipts.jsonl` is written correctly.
-   Verify `agent why` command retrieves correct reasoning.
