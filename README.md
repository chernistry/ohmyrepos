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

Oh My Repos indexes your GitHub repositories, generates LLM summaries and embeddings, and exposes a hybrid semantic + lexical search engine plus a streaming RAG chat interface.

The project consists of:

- **FastAPI backend** – `/api/v1/search` and `/api/v1/chat` over Qdrant + BM25.
- **Next.js frontend** – interactive search grid and AI chat UI.
- **CLI pipeline** – collection, summarization and embedding of repositories.

---

## Key Features

- **Hybrid search engine** combining BM25 (BM25Plus) with dense vector similarity (Jina embeddings) and optional LLM-based reranking.
- **RAG-powered chat** that answers questions about your repositories using retrieved context, streamed over SSE.
- **Async ingestion pipeline** that collects starred repositories from GitHub, summarizes them with LLMs, and stores embeddings in Qdrant.
- **Pluggable LLM stack** with OpenAI/OpenRouter-compatible APIs or local Ollama, plus Jina-based embeddings and reranker.
- **Production-oriented architecture** with structured logging, metrics hooks and clear separation between API, UI and batch ingestion.

---

## Quick Start

> [!NOTE]
> This quick start assumes you already have API keys for GitHub, a Qdrant instance (local or cloud), and embedding/LLM providers (e.g. Jina + OpenRouter/OpenAI). See `docs/IMPLEMENTATION.md` for full configuration details.

> [!WARNING]
> Indexing large collections of repositories uses paid LLM and embedding APIs. Start with a small subset while validating your setup.

### 1. Clone and install backend

```bash
git clone https://github.com/chernistry/ohmyrepos.git
cd ohmyrepos

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` from template and edit the required values:

```bash
cp .env.example .env
```

Set at least:

- `GITHUB_USERNAME`, `GITHUB_TOKEN` – GitHub user and personal access token.
- `QDRANT_URL`, `QDRANT_API_KEY` – Qdrant instance (cloud or local).
- `EMBEDDING_MODEL_API_KEY` – embedding + reranker provider key (e.g. Jina).
- `CHAT_LLM_PROVIDER`, `CHAT_LLM_BASE_URL`, `CHAT_LLM_MODEL`, `CHAT_LLM_API_KEY` – LLM provider configuration.

### 3. Start backend API

```bash
./run.sh b start
# FastAPI will listen on http://127.0.0.1:8000
```

### 4. Start Next.js frontend

```bash
cd ui
npm install
npm run dev
# or from repo root:
# ./run.sh f start
```

Open the UI:

- Search & chat: `http://localhost:3000`

### 5. (Optional) Ingest your repositories via CLI

Once environment and Qdrant are configured, you can index your own repositories using the CLI:

```bash
# Collect starred repositories
python ohmyrepos.py collect --output repos.json --incremental

# Summarize and embed repositories, then push to Qdrant
python ohmyrepos.py embed --input repos.json --skip-collection \
  --concurrency 4 --output enriched_repos.json
```

The Next.js UI will then query the hybrid search API backed by your Qdrant collection.

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

## Architecture (High Level)

The system exposes a single user-facing surface (Next.js UI) that talks to the FastAPI backend. The backend coordinates BM25 search over `repos.json`, vector search over Qdrant, and optionally LLMs for summarization, reranking and chat.

```mermaid
graph LR
  User[Browser] --> UI[Next.js UI (ui/)]
  UI -->|/api/v1/search,/chat| API[FastAPI API]
  API --> Search[HybridRetriever]
  Search --> Qdrant[(Qdrant Vector DB)]
  Search --> BM25[BM25 over repos.json]
  API --> Chat[Chat / RAG Engine]
  Chat --> LLM[LLMGenerator (OpenAI/OpenRouter/Ollama)]
  LLM --> Qdrant
```

Key flows:

- **Search** – UI sends queries to `/api/v1/search`, FastAPI uses `HybridRetriever` to combine BM25 + vector search and returns normalized results.
- **Chat** – UI streams from `/api/v1/chat` via SSE; the backend performs a short RAG retrieval step, then streams LLM output chunk-by-chunk as `data: {"chunk": "..."}\n\n`.
- **Ingestion** – CLI commands (`collect`, `summarize`, `embed`, `ingest`) run the GitHub → LLM → embeddings → Qdrant pipeline in batch mode.

---

## Additional Details

<details>
<summary>CLI commands (summary)</summary>

- `python ohmyrepos.py collect` – fetch starred repositories from GitHub.
- `python ohmyrepos.py summarize` – LLM-based summarization for repositories in a JSON file.
- `python ohmyrepos.py embed` – full pipeline: collect/summarize/embed/store.
- `python ohmyrepos.py search` – run hybrid search from the terminal.
- `python ohmyrepos.py ingest` / `reindex` – ingest and reindex repositories into Qdrant.

Full CLI and pipeline details are documented in `docs/IMPLEMENTATION.md`.

</details>

<details>
<summary>Local Qdrant and Ollama (optional)</summary>

- A `docker-compose.yml` is provided to run:
  - local Qdrant (`qdrant/qdrant`) on `localhost:6333`,
  - local Ollama (`ollama/ollama`) on `localhost:11434`.
- For local-only setups, set:
  - `QDRANT_URL=http://localhost:6333`,
  - `CHAT_LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://127.0.0.1:11434`.

</details>

---

📚 [Detailed Implementation Guide](docs/IMPLEMENTATION.md)

