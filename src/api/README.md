# Oh My Repos API

FastAPI-based search API for Oh My Repos.

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

### Health Checks

- `GET /healthz` - Lightweight health check
- `GET /readyz` - Readiness check with dependency validation

### Search

- `POST /api/v1/search` - Search repositories

**Request:**
```json
{
  "query": "machine learning python",
  "limit": 25,
  "offset": 0,
  "filter_tags": ["python", "ml"]
}
```

**Response:**
```json
{
  "query": "machine learning python",
  "results": [
    {
      "repo_name": "owner/repo",
      "full_name": "owner/repo",
      "description": "Repository description",
      "summary": "AI-generated summary",
      "tags": ["python", "ml", "tensorflow"],
      "language": "Python",
      "stars": 1234,
      "url": "https://github.com/owner/repo",
      "score": 0.95
    }
  ],
  "total": 1
}
```

## Docker

```bash
# Build
docker build -f Dockerfile.api -t ohmyrepos-api .

# Run
docker run -p 8000:8000 --env-file .env ohmyrepos-api
```

## Configuration

See `.env.example` for required environment variables.

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
