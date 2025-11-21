# Ingestion Pipeline

Modernized data ingestion pipeline for Oh My Repos.

## Overview

The ingestion pipeline handles:
- Fetching repository metadata from GitHub API
- Extracting README content
- LLM-powered summarization
- Storing enriched data in Qdrant

## Usage

### Ingest Single Repository

```bash
python ohmyrepos.py ingest https://github.com/owner/repo
```

Options:
- `--output`, `-o`: Save ingested data to JSON file

### Reindex from File

```bash
python ohmyrepos.py reindex repos.json
```

This command:
1. Loads repositories from JSON file
2. Summarizes any repos without summaries
3. Stores all repos in Qdrant

## Architecture

```
┌─────────────────┐
│  GitHub API     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fetch Metadata │
│  + README       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Summarizer │
│  (OpenRouter/   │
│   Ollama)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Qdrant Store   │
│  (Vector DB)    │
└─────────────────┘
```

## Pipeline Class

```python
from src.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()
await pipeline.initialize()

# Ingest single repo
result = await pipeline.ingest_repo("https://github.com/owner/repo")

# Reindex from file
results = await pipeline.reindex(Path("repos.json"))

await pipeline.close()
```

## Data Flow

1. **Fetch**: Get repository metadata and README from GitHub
2. **Enrich**: Generate summary and tags using LLM
3. **Store**: Upsert to Qdrant with embeddings

## Error Handling

- Invalid URLs raise `ValueError`
- GitHub API errors are propagated
- Missing configuration raises `ValueError`
- Network errors are logged and raised

## Testing

```bash
pytest tests/test_ingestion.py -v
```

## Configuration

Required environment variables:
- `GITHUB_TOKEN`: GitHub personal access token
- `GITHUB_USERNAME`: GitHub username
- `QDRANT_URL`: Qdrant server URL
- `QDRANT_API_KEY`: Qdrant API key (optional for local)
- `CHAT_LLM_API_KEY`: LLM provider API key

See `.env.example` for full configuration.
