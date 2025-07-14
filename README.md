# Oh My Repos

## Overview
Oh My Repos is an internal tool for semantic search and analysis of GitHub starred repositories. It provides both a Streamlit-based web UI and a Typer-powered CLI that leverage hybrid retrieval, reranking, and large-language-model (LLM) summarisation to surface the most relevant repositories for a given query.

## Project Structure
```
├── ohmyrepos.py           # Thin CLI wrapper
├── requirements.txt       # Python package requirements
├── src/                   # Application source code
│   ├── app.py             # Streamlit UI entry-point
│   ├── cli.py             # Typer CLI entry-point
│   ├── config.py          # Centralised settings (env-driven)
│   ├── core/              # Retrieval & storage layer
│   │   ├── collector.py   # GitHub data ingestion
│   │   ├── retriever.py   # Hybrid BM25 + vector retrieval
│   │   ├── reranker.py    # Jina AI reranking wrapper
│   │   ├── storage.py     # Qdrant vector-store integration
│   │   ├── summarizer.py  # LLM-based summarisation
│   │   └── embeddings/    # Embedding provider factory & drivers
│   └── llm/               # LLM orchestration utilities
│       ├── chat_adapter.py
│       ├── generator.py
│       ├── prompt_builder.py
│       └── providers/     # OpenAI & Ollama adapters
├── tests/                 # Pytest suite
├── prompts/               # Prompt templates
├── .env-example           # Sample environment variables
└── repos.json             # Cached GitHub metadata
```

## Prerequisites
* Python ≥ 3.9
* Git
* (Optional) [Qdrant](https://qdrant.tech/) instance for vector storage
* (Optional) Local Ollama server **or** valid API keys for OpenAI/Jina AI

## Installation
```bash
# 1. Clone the repo
$ git clone <corp-git-url>/ohmyrepos.git && cd ohmyrepos

# 2. Create & activate a virtual environment
$ python -m venv .venv && source .venv/bin/activate

# 3. Install Python dependencies
$ pip install -r requirements.txt

# 4. Copy environment template & edit values
$ cp .env-example .env
$ $EDITOR .env
```

## Configuration
The application is configured via environment variables parsed by `pydantic_settings`. Key parameters:

| Variable | Purpose |
|----------|---------|
| `GITHUB_USERNAME` / `GITHUB_TOKEN` | GitHub API access for repository metadata |
| `CHAT_LLM_PROVIDER` | `openai` or `ollama` |
| `CHAT_LLM_API_KEY`  | API key if using remote provider |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | Local Ollama endpoint and model name |
| `QDRANT_URL` / `QDRANT_API_KEY`    | Vector-store connection |
| `EMBEDDING_MODEL` / `EMBEDDING_MODEL_API_KEY` | Jina embedding model |
| `RERANKER_MODEL` | Jina reranker model |

Refer to `.env-example` for the full list.

## Usage
### CLI
```bash
# Index starred repositories
$ python ohmyrepos.py ingest --user $GITHUB_USERNAME

# Semantic search from terminal
$ python ohmyrepos.py search "vector databases"
```

### Streamlit UI
```bash
$ streamlit run src/app.py
```

## Current Status / Work in Progress
* Retrieval, reranking, and Streamlit UI are functional.
* Upcoming: Embedding provider fallbacks, LangGraph integration, advanced retrieval pipelines.

## License
Licensed under the Creative Commons Zero v1.0 Universal license (CC0-1.0). See [LICENSE](LICENSE) for details.

## Acknowledgments
This project builds on open-source packages including Typer, Streamlit, Qdrant, and Jina AI.

## References
* Qdrant Docs – https://qdrant.tech/documentation/
* Jina AI Embeddings – https://github.com/jina-ai
* Streamlit – https://docs.streamlit.io/
* Typer – https://typer.tiangolo.com/

<!-- Future sections: Embedding Strategies, Retrieval Pipeline, LangGraph Integration -->
