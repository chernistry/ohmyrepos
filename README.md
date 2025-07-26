# Oh My Repos

## Overview
Oh My Repos is a robust semantic search and analysis solution for GitHub starred repositories. It offers both a Streamlit-based web user interface and a comprehensive command-line interface (CLI). The system leverages hybrid retrieval (BM25 and vector search), advanced reranking, and large-language model (LLM) summarization to identify and present the most relevant repositories in response to user queries.

## Key Capabilities
- **Hybrid Search**: Integrates BM25 keyword matching with vector similarity for comprehensive results.
- **Intelligent Reranking**: Employs Jina AI models to enhance result precision.
- **Automated Summarization**: LLM-driven analysis generates concise summaries and relevant tags for repositories.
- **Dual Interface Support**: Provides a CLI for scripting and automation, alongside a Streamlit UI for interactive exploration.
- **Configurable LLM Backends**: Supports various LLM providers, including OpenAI, OpenRouter, and local Ollama deployments.

## Architectural Structure
```
├── ohmyrepos.py           # CLI entry point wrapper
├── requirements.txt       # Python dependency manifest  
├── .env-example           # Environmental configuration template
├── repos.json             # Cached repository metadata store
├── debug/                 # Artifacts and results from debugging sessions
├── prompts/               # LLM prompt templates
│   └── summarize_repo.md  # Template for repository summarization
├── src/                   # Core application source code
│   ├── app.py             # Streamlit web interface entry point
│   ├── cli.py             # Typer CLI implementation
│   ├── config.py          # Centralized, environment-driven settings management
│   ├── core/              # Data processing and retrieval layer
│   │   ├── collector.py   # GitHub API data ingestion
│   │   ├── retriever.py   # Hybrid search engine logic
│   │   ├── reranker.py    # Jina AI reranking integration
│   │   ├── storage.py     # Qdrant vector database integration
│   │   ├── summarizer.py  # LLM-based summarization logic
│   │   ├── clusterer.py   # Placeholder for future repository clustering
│   │   └── embeddings/    # Embedding provider abstraction
│   │       ├── base.py    # Base class for embedding providers
│   │       ├── factory.py # Factory for dynamic embedding provider selection
│   │       └── providers/jina.py # Jina AI embedding provider
│   └── llm/               # LLM orchestration utilities
│       ├── chat_adapter.py    # Adapts chat interactions for LLMs
│       ├── generator.py       # Manages LLM text generation
│       ├── prompt_builder.py  # Constructs LLM prompts
│       ├── reply_extractor.py # Extracts structured replies from LLM responses
│       └── providers/         # LLM backend integrations
│           ├── base.py        # Base class for LLM providers
│           ├── openai.py      # OpenAI LLM integration
│           └── ollama.py      # Ollama LLM integration
└── tests/                 # Unit and integration test suite
    ├── test_collector.py      # Tests for the repository collector
    └── test_ollama_provider.py # Tests for the Ollama LLM provider
```

## Getting Started

### Prerequisites
- **Python**: Version 3.9 or higher.
- **Git**: Distributed version control system.
- **GitHub Personal Access Token**: Required for accessing GitHub repository data.
- **Optional Components**:
    - **Qdrant Instance**: For vector database persistence.
    - **LLM API Keys**: Required for remote LLM providers (e.g., OpenAI, OpenRouter).
    - **Local Ollama Server**: For local LLM inference.

### Installation
```bash
# Clone the repository
git clone https://github.com/chernistry/ohmyrepos && cd ohmyrepos

# Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Copy the environment template and configure
cp .env-example .env
# Open .env in a text editor to set your credentials and configurations
```

### Configuration
The application's behavior is managed via environment variables, parsed by `pydantic_settings`. A subset of critical parameters is listed below; refer to `.env-example` for a complete reference.

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `GITHUB_USERNAME` / `GITHUB_TOKEN` | GitHub API authentication credentials | `your_username` / `ghp_xxx_your_token` |
| `CHAT_LLM_PROVIDER` | Specifies the LLM backend to utilize | `openai` \| `ollama` |
| `CHAT_LLM_API_KEY` | API key for remote LLM services | `sk-xxxxxxxxx` (OpenAI/OpenRouter) |
| `EMBEDDING_MODEL_API_KEY` | API key for Jina AI embedding service | `jina_xxx_your_key` |
| `QDRANT_URL` / `QDRANT_API_KEY` | Connection details for the Qdrant vector database | `https://qdrant.cloud` / `your_qdrant_api_key` |

## Operational Usage

### CLI Commands
The Typer-based CLI provides the following functionalities:

```bash
# Execute the full pipeline: collect → summarize → embed repositories
python ohmyrepos.py embed

# Collect starred repositories from GitHub and save to a JSON file
python ohmyrepos.py collect --output repos.json

# Generate LLM-powered summaries for collected repositories
python ohmyrepos.py summarize repos.json --output summaries.json

# Perform a semantic search on indexed repositories
python ohmyrepos.py search "vector databases" --limit 10

# Launch the Streamlit web interface
python ohmyrepos.py serve --port 8501
```

### Streamlit UI
To launch the interactive web interface:

```bash
streamlit run src/app.py
# Access the UI via your web browser, typically at: http://localhost:8501
```

### Advanced Operations
```bash
# Process repository summaries with a specified level of concurrency
python ohmyrepos.py summarize repos.json --concurrency 4

# Generate embeddings for an existing set of summarized repositories
python ohmyrepos.py embed-only --input summaries.json

# Conduct a filtered semantic search using specific tags
python ohmyrepos.py search "machine learning" --filter-tags python,tensorflow
```

## Functional Overview

The system operates through a series of integrated stages:

1.  **Collection**: Retrieves comprehensive metadata for starred repositories via the GitHub API.
2.  **Summarization**: An LLM analyzes repository READMEs and descriptions to generate concise summaries and extract relevant tags.
3.  **Embedding**: Transforms textual data into high-dimensional vector representations suitable for semantic search.
4.  **Storage**: Persists the generated embeddings within a Qdrant vector database for efficient retrieval.
5.  **Retrieval**: Executes hybrid searches, combining BM25 keyword relevance with vector similarity, further refined by AI reranking for optimal results.

## Development Guidelines

### Test Execution
Unit and integration tests can be executed using `pytest`:

```bash
pytest tests/ -v
```

### Code Quality Assurance
Code formatting and linting are enforced using `black` and `ruff`:

```bash
black src/ tests/
ruff check src/ tests/
```

## Future Enhancements
-   **Repository Clustering**: Implementation of algorithms for automatic, topic-based grouping of repositories.
-   **Enhanced Embedding Robustness**: Development of multi-provider fallback strategies for embedding generation.
-   **Advanced Analytics**: Introduction of features for in-depth repository similarity scoring and trend analysis.
-   **Export Capabilities**: Addition of functionalities for exporting search results and summaries in various formats (e.g., Markdown, CSV).

## License
This project is licensed under the Creative Commons Zero v1.0 Universal (CC0-1.0) license. For detailed terms, refer to the [LICENSE](LICENSE) file.

## Acknowledgments
This project leverages the following open-source technologies:
-   **Typer**: For building the command-line interface.
-   **Streamlit**: For developing the interactive web user interface.
-   **Qdrant**: As the high-performance vector database.
-   **Jina AI**: For advanced embedding and reranking capabilities.
-   **OpenAI**: For large-language model integration.
