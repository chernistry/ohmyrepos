# Ticket 04: Ingestion Pipeline Refresh

**Goal**: Modernize the data ingestion pipeline to support the new architecture (Qdrant Cloud + BGE-M3).

## Context
The current ingestion logic is likely tied to the local Streamlit/Qdrant setup. We need a standalone script/CLI that can run locally or in a background job to update the index.

## Requirements
- [ ] **Ingestion Logic Refactor**:
    - Create `src/ingestion/` module.
    - Implement `ingest_repo(repo_url)`:
        - Fetch metadata from GitHub.
        - Clone/fetch code (shallow).
        - Chunk content (README, code files).
- [ ] **Embedding Pipeline**:
    - Integrate `BGE-M3` (via `FlagEmbedding` or `sentence-transformers` or `Ollama`).
    - Ensure it produces both dense and sparse vectors (if using BGE-M3's native sparse) or use Qdrant's sparse vector support.
    - *Decision*: Use BGE-M3 for dense. For sparse, check if we use BGE-M3's sparse output or Qdrant's BM25. (Refer to `best_practices.md`: "BGE-M3... dense + sparse").
- [ ] **Qdrant Upsert**:
    - Update `src/core/storage.py` (or new module) to handle the new schema (dense + sparse vectors).
    - Ensure upsert is batched and robust.
- [ ] **CLI**:
    - Add `ohmyrepos ingest <url>` command.
    - Add `ohmyrepos reindex` command (iterates over `repos.json`).

## Acceptance Criteria
- `ohmyrepos ingest https://github.com/owner/repo` works and populates Qdrant Cloud.
- Data in Qdrant includes correct payload and vectors.
- Search API (Ticket 02) can find the ingested repo.
