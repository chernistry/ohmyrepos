# Scripts

Utility scripts for Oh My Repos.

## verify_connections.py

Verifies connectivity to all external services required by the application.

**Usage:**
```bash
python scripts/verify_connections.py
```

**Checks:**
- GitHub API (authentication and access)
- Qdrant Vector Database (connection and collections)
- OpenRouter/LLM API (authentication and model availability)
- Jina Embeddings API (authentication and model access)
- Ollama (optional, for local LLM)

**Exit codes:**
- `0`: All critical services accessible
- `1`: One or more critical services failed (Ollama failures are ignored)

**Requirements:**
- Valid `.env` file with all required credentials
- Network connectivity to external services
