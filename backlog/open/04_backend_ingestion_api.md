# Ticket 04: Backend Ingestion API (Incremental)

**Goal**: Implement the backend endpoints to support the UI's sync features, focusing on incremental updates.

## Context
The UI needs endpoints to trigger syncs and check status. The core logic must be smart enough to skip repos that haven't changed.

## Requirements

### 1. Data Model Updates
- [ ] Add `last_indexed_at` and `last_pushed_at` (from GitHub) to Qdrant payload.
- [ ] Store `sync_status` (running/idle) in memory or a simple file/db.

### 2. API Endpoints (`src/api/ingest.py`)
- [ ] `POST /api/v1/ingest`: Accepts `{ "url": "..." }`.
- [ ] `POST /api/v1/sync`: Triggers full sync of all tracked repos.
- [ ] `GET /api/v1/sync/status`: Returns `{ "status": "running", "progress": 50, "total": 100 }`.

### 3. Incremental Logic (`src/core/ingestion.py`)
- [ ] **Logic**:
  1. Fetch repo metadata from GitHub API.
  2. Compare `github.pushed_at` with `qdrant.payload.last_pushed_at`.
  3. If `github > qdrant` (or if not in Qdrant), proceed to clone & embed.
  4. Else, skip.
- [ ] **Concurrency**: Use `asyncio` or a background task queue (FastAPI `BackgroundTasks` is fine for MVP).

## Implementation Snippets

**`src/api/ingest.py`**:
```python
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from src.core.ingestion import run_incremental_sync

router = APIRouter()

class IngestRequest(BaseModel):
    url: str

@router.post("/sync")
async def trigger_sync(background_tasks: BackgroundTasks):
    # Offload to background so UI doesn't hang
    background_tasks.add_task(run_incremental_sync)
    return {"status": "started"}
```

**`src/core/ingestion.py` (Incremental Check)**:
```python
async def should_reindex(repo_id: str, github_pushed_at: str) -> bool:
    # Fetch current payload from Qdrant
    points = client.retrieve(collection_name="repos", ids=[repo_id])
    if not points:
        return True
    
    stored_pushed_at = points[0].payload.get("last_pushed_at")
    if not stored_pushed_at:
        return True
        
    # Compare dates (using dateutil or similar)
    return parse(github_pushed_at) > parse(stored_pushed_at)
```

## Acceptance Criteria
- `/sync` returns immediately and starts background work.
- Logic correctly skips repos that haven't changed (verify via logs).
- `last_pushed_at` is updated in Qdrant after successful ingest.
