# Ticket 02: FastAPI Search API

**Goal**: Implement the core Search API using FastAPI, wrapping the existing or new retrieval logic.

## Context
The backend will be a stateless FastAPI service deployed on Render. It needs to expose search and health endpoints.

## Requirements
- [ ] **FastAPI Setup**:
    - Create `src/api/main.py` as the entrypoint.
    - Configure CORS (allow localhost:3000 for dev, and production Vercel URL).
- [ ] **Endpoints**:
    - `GET /healthz`: Returns `{"status": "ok"}`.
    - `GET /readyz`: Checks Qdrant connection.
    - `POST /api/v1/search`:
        - Input: `SearchRequest` (query, filters, limit, offset).
        - Output: `SearchResponse` (list of repos).
        - Logic: Call `src.core.retriever` (or equivalent) to perform hybrid search.
- [ ] **Refactoring**:
    - Ensure `src/core` logic is decoupled from Streamlit.
    - If `src/core/retriever.py` is tightly coupled, refactor it to accept a Qdrant client instance or config.
- [ ] **Dockerfile**:
    - Create `Dockerfile` for the API service.
    - Ensure it installs dependencies and runs `uvicorn`.

## Acceptance Criteria
- API starts locally with `uvicorn src.api.main:app --reload`.
- `/healthz` returns 200.
- `/api/v1/search` returns results (mocked or real if Qdrant is populated).
- Docker build succeeds.
