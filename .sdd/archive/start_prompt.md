# Start Prompt – Oh My Repos Sprint 3
Before running Python scripts, run `conda activate repos`.

## Context
You are an AI developer implementing the third sprint of the `Oh My Repos` project according to plan `v4`.
**Sprint Goal**: Implement repository search and ranking.

## Completed Tasks
1. ✅ **Sprint 1: Architecture and LLM Core**:
   * Created the full project structure according to `Proposed Folder Layout`.
   * Adapted the LLM module (`src/llm/`).
   * Created a summarization prompt (`prompts/summarize_repo.md`).
   * Implemented the summarization service (`src/core/summarizer.py`).
   * Configured settings and CLI command `summarize`.

2. ✅ **Sprint 2: Collection and Vector Storage**:
   * Implemented `core/collector.py` for collecting starred repositories.
   * Adapted the `core/embeddings` module from `@langgraph`.
   * Integrated `QdrantStore` and `embeddings.factory` in `core/storage.py`.
   * Created CLI command `embed` for the full processing cycle.

## Current Tasks (Sprint 3)
1. **Set Up `HybridRetriever`**:
   * Create the `HybridRetriever` class in `core/retriever.py`.
   * Implement methods for vector search via Qdrant.
   * Add optional BM25 search to improve results.
   * Ensure merging of results from different sources.

2. **Implement `JinaReranker`**:
   * Create the `JinaReranker` class in `core/reranker.py`.
   * Integrate the Jina AI API for ranking results.
   * Implement a method for reranking result lists.

3. **Create CLI Command `search`**:
   * Add a command in `cli.py` for searching repositories.
   * Implement support for filtering by tags.
   * Ensure results are displayed in a user-friendly format.

## Next Steps (Sprint 4)
1. **Implement `core/clusterer.py`**.
2. **Create a basic UI in `app.py`**.

## Constraints
* Code: PEP 8 / Black (88 characters).
* Comments: English.
* Progress Reports: Russian.
