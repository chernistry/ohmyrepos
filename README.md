# Oh My Repos

Semantic search and RAG chat over your GitHub repositories, built on a hybrid BM25 + vector search stack with a modern FastAPI + Next.js architecture.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/UI-Next.js-black)](https://nextjs.org/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-35bfa4)](https://qdrant.tech/)
[![Jina](https://img.shields.io/badge/Embeddings-Jina_AI-4b3aff)](https://jina.ai/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)

---

## Overview

Oh My Repos indexes your GitHub repositories, generates summaries/embeddings, and serves hybrid search plus streaming RAG chat.

![Oh My Repos Dashboard](assets/screenshot.png)

---

## Key Features

- Hybrid search (BM25 + dense vectors) with optional reranking.
- Streaming RAG chat over SSE using retrieved repo context.
- Async ingestion pipeline for GitHub stars → summaries → embeddings → Qdrant.
- Pluggable providers: OpenRouter/OpenAI/Ollama for LLM; Jina or Ollama embeddings.
- Scripts for local dev, Docker stack, and MCP tools (Claude/Cursor).

---

## Quick Start

1) Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ui && npm install && cd ..
```

2) Configure & ingest (auto-prompts, defaults to Ollama embeddings with 10s timeout)
```bash
cp .env.example .env
./run.sh setup   # collects keys, pulls embedding model if needed, runs collect+embed
```

3) Run locally
```bash
./run.sh dev           # backend http://127.0.0.1:8000 + frontend http://localhost:3000
# or ./run.sh b start / ./run.sh f start to run individually
```

4) Docker stack (API + Qdrant only; Ollama stays on host)
```bash
./run.sh stack up
./run.sh stack down
./run.sh stack logs    # tail API/Qdrant logs
```

5) MCP tools (Claude Desktop / Cursor)
```bash
./run.sh mcp           # stdio transport for MCP tools
```

---

## Tech Stack

| Layer        | Technology                                       |
|-------------|---------------------------------------------------|
| Backend API | FastAPI, httpx, Pydantic, structlog               |
| Frontend UI | Next.js (App Router), React, Tailwind CSS         |
| Vector DB   | Qdrant (cloud or Docker)                          |
| Embeddings  | Jina embeddings (default) or Ollama embeddings    |
| Reranker    | Jina reranker                                     |
| LLM         | OpenAI/OpenRouter-compatible APIs or local Ollama |
| CLI / Batch | Typer, Rich                                       |

---

📚 [Detailed Implementation Guide](docs/IMPLEMENTATION.md)
