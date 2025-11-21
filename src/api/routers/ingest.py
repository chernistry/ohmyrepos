from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from src.ingestion.pipeline import IngestionPipeline

router = APIRouter()

class IngestRequest(BaseModel):
    url: str
    force: bool = False

async def run_ingestion_task(url: str, force: bool):
    pipeline = IngestionPipeline()
    await pipeline.initialize()
    try:
        await pipeline.ingest_repo(url, force=force)
    finally:
        await pipeline.close()

@router.post("/ingest")
async def ingest_repo(request: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger ingestion for a repository."""
    background_tasks.add_task(run_ingestion_task, request.url, request.force)
    return {"status": "accepted", "message": f"Ingestion started for {request.url}"}

@router.post("/sync")
async def sync_all(background_tasks: BackgroundTasks):
    """Trigger sync for all tracked repositories (placeholder)."""
    # TODO: Implement full sync logic (iterate over all repos in DB)
    return {"status": "not_implemented", "message": "Full sync not yet implemented"}
