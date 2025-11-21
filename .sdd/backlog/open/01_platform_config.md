# Ticket 01: Platform & Config Setup

**Goal**: Prepare the repository structure, configuration management, and external service connections for the modernized Oh My Repos.

## Context
We are moving from a local Streamlit app to a hosted Next.js + FastAPI architecture. This ticket establishes the foundation.

## Requirements
- [ ] **Repository Structure**:
    - Ensure `src/api` exists for FastAPI.
    - Ensure `ui/` (or `src/ui`) exists for Next.js (initialize if needed in Ticket 03, but reserve the path).
    - Ensure `src/config` exists.
- [ ] **Configuration Module (`src/config`)**:
    - Create a robust configuration loader (using `pydantic-settings` or similar).
    - Define variables for:
        - `QDRANT_URL`, `QDRANT_API_KEY`
        - `OPENROUTER_API_KEY`
        - `GITHUB_TOKEN` (for ingestion)
        - `LOG_LEVEL`
- [ ] **Environment Variables**:
    - Create `.env.example` with all required keys.
    - Add `.env` to `.gitignore` (if not already there).
- [ ] **Service Verification Scripts**:
    - Create a simple script `scripts/verify_connections.py` to:
        - Connect to Qdrant Cloud and list collections.
        - Call OpenRouter API (simple "hello" check).
        - Check GitHub API access.

## Acceptance Criteria
- `src/config` module is importable and loads env vars.
- `scripts/verify_connections.py` runs successfully with valid credentials.
- Repo structure is clean and ready for API/UI work.
