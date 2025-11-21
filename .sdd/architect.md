# Oh My Repos — Architecture & Implementation Plan (2025)

This document is the source of truth for modernising Oh My Repos from a local Streamlit/BM25+Qdrant tool into a 2025‑ready, low‑cost online service.

Context sources:
- Project description: `.sdd/project.md`
- Research & best practices: `.sdd/best_practices.md`
- Reference repos: `.sdd/ref/**`

Target stack (from research):
- Backend API: Python 3.11+, FastAPI, `qdrant-client`, `httpx`
- Frontend: Next.js 14 (App Router) on Vercel Hobby
- Vector store: Qdrant Cloud free tier (hybrid dense+sparse)
- LLMs: OpenRouter (plus optional local Llama via Ollama)
- Embeddings: BGE‑M3 (local) + optional Jina embeddings API

---

## Hard Constraints
- Zero/near‑zero infra budget; prefer free tiers and local compute.
- No long‑lived GitHub PATs with wide scopes; GitHub OAuth and least privilege for hosted version.
- No private repositories in v1 (public only) — avoids complex access control and data protection risk.
- Personal data: avoid ingesting PII; minimal logging of user queries (hashed or truncated where feasible).
- Vendor lock‑in: abstraction layer around LLM providers and embeddings (OpenRouter/OpenAI/Ollama/Jina).
- All network calls must have timeouts, retries with backoff+jitter, and explicit error handling.
- Self‑hostable locally without cloud dependencies (via local Qdrant and Ollama) for power users.

## Go/No-Go Preconditions
- GitHub app or OAuth app configured with repo read scope for public repos only.
- Qdrant Cloud free cluster provisioned (or local Qdrant for dev) with collections sized for target repo count.
- OpenRouter API key (and optionally Ollama running locally) available and configured.
- Central config for:
  - Qdrant URL/API key, collection names.
  - LLM providers/keys, models, and rate limits.
  - GitHub API credentials and rate‑limit settings.
- Monitoring access:
  - Ability to inspect logs (Render/Vercel logging + local structured logs).
  - Basic uptime checks or ping endpoint.
- Seed dataset agreed (e.g. current `repos.json`/`enriched_repos.json`) to validate migration.

Go if:
- Secrets exist and are stored securely (env vars / secret manager).
- Free‑tier quotas are sufficient for target usage.

No‑Go / pause if:
- Qdrant or GitHub rate limits cannot be respected.
- LLM quotas are insufficient and no fallback/local model is configured.

## Goals & Non‑Goals
**Goals**
- Hosted version of Oh My Repos with modern web UI and shareable URL.
- Hybrid search (lexical + dense) with high relevance over GitHub repo collections.
- Cost‑aware usage of LLMs (reranking/summarisation only where they add value).
- Clean separation between ingestion/indexing, search API, and UI.
- Clear observability: basic metrics, logs, and health checks.

**Non‑Goals (v1)**
- No private repo ingestion or complex multi‑tenant ACLs.
- No full multi‑region HA or strict SLO enforcement (best‑effort on free tiers).
- No full self‑service account system (simple single‑user or light multi‑user).
- No heavy workflow/orchestration platform (Airflow, Prefect) — simple jobs/CLIs are enough.

## Alternatives
### A) Split UI (Next.js) + API (FastAPI) + Qdrant Cloud (Recommended)
- Pros:
  - Matches research patterns and free‑tier sweet spots (Vercel + Render + Qdrant Cloud).
  - Clear separation of concerns; easy to evolve UI and API independently.
  - Uses current Python core modules with minimal refactor into FastAPI.
- Cons:
  - Two deploy targets (Vercel + Render/Railway).
  - Cold starts on free tiers; extra work around timeouts and retries.

### B) Single FastAPI app serving both API and server‑rendered UI
- Pros:
  - One deploy target (Render/Railway), simpler infra.
  - Easier local dev (single service).
- Cons:
  - Less modern UX vs. Next.js; no Vercel edge features.
  - Less aligned with long‑term goal of rich web UI.

### C) Keep Streamlit UI + Add Thin API
- Pros:
  - Minimal change; reuse existing UI.
- Cons:
  - Streamlit not ideal for hosted, multi‑user, modern UX.
  - Harder to integrate advanced flows (RAG assistant, interactive filters).

## MVP Recommendation
- Choose **Alternative A**: Next.js frontend on Vercel + FastAPI backend on Render/Railway + Qdrant Cloud hybrid search.
- Migration path:
  - Reuse existing Python core modules (`src/core/*`, `src/llm/*`) inside a new FastAPI app.
  - Extract search and ingestion flows into explicit API endpoints and background jobs.
  - Implement minimal Next.js UI that consumes these APIs and mirrors current CLI/search features.
- Rollback plan:
  - Keep existing local CLI and Streamlit flow operational until web MVP is stable.
  - If costs or complexity spike, fallback to single FastAPI+Jinja app or purely local mode.

## Architecture Overview
High‑level components:
- **Ingestion Job** (Python CLI / script)
  - Reads GitHub repos from `repos.json` or GitHub API.
  - Uses LLM summariser (`src/core/summarizer.py` + `src/llm/*`) to generate structured summaries.
  - Uses embedding factory (`src/core/embeddings/*`) to produce BGE‑M3 vectors (local) and optional Jina vectors.
  - Upserts documents into Qdrant (hybrid dense+sparse) with rich metadata.
- **Search API** (FastAPI)
  - Endpoints for search, repo details, tags/facets, health check.
  - Wraps `src/core/retriever.py`, `src/core/reranker.py`, `src/core/storage.py`.
  - Applies hybrid search (dense + sparse/BM25) and optional LLM reranking.
- **UI** (Next.js)
  - Search page with filters and results list.
  - “Ask the repos” assistant panel that calls a RAG endpoint.
  - Basic settings view (providers, limits, local/remote toggles).
- **Config & Secrets**
  - Centralised in `src/config` (Python) and `.env`/env vars for both API and UI.
- **Monitoring & Logging**
  - `src/core/logging.py` and `src/core/monitoring.py` for structured logs and metrics.
  - Basic counters: request counts, latencies, error rates, LLM call counts, Qdrant query counts.

Data flow:
1. Ingestion job processes repos and writes enriched docs + vectors to Qdrant.
2. UI sends queries to `/api/search`.
3. API runs hybrid query in Qdrant, optionally reranks via LLM, returns results.
4. UI displays results with facets/tags; optional RAG assistant uses `/api/ask`.

## Discovery Notes
Repository structure (partial):
- `src/core/collector.py` — GitHub collection pipeline (starred repos, metadata).
- `src/core/summarizer.py` — LLM‑based repo summarisation.
- `src/core/storage.py` — Qdrant integration and vector storage.
- `src/core/retriever.py` — hybrid retrieval logic.
- `src/core/reranker.py` — reranking via Jina/LLM.
- `src/llm/*` — LLM provider abstraction (OpenAI, Ollama).
- `src/app.py`, `src/cli.py` — legacy Streamlit app and CLI.

Extension points:
- Extraction of search logic into FastAPI router modules.
- Reuse of config module for environment‑specific settings.
- Adding monitoring hooks (metrics/logging) into core flows.

## MCDM Summary for UI/API Architecture
Criteria: PerfGain, SecRisk, DevTime, Maintainability, Cost, Scalability, DX.

Alternative A (Next.js + FastAPI + Qdrant Cloud):
- High DX, good scalability, moderate dev time, low infra cost using free tiers.

Alternative B (FastAPI + server‑rendered UI):
- Simpler infra, slightly lower DX and UX, similar cost.

Alternative C (Streamlit + thin API):
- Lowest dev time short‑term, but poor long‑term maintainability and UX.

Rank (TOPSIS‑style, informally):
1. **A** — best overall balance for long‑term project.
2. B — viable fallback.
3. C — not recommended except as temporary bridge.

## Key Decisions (ADR‑Style)
- **ADR‑001 — Adopt Next.js + FastAPI + Qdrant Cloud**
  - Decision: Use Next.js 14 (App Router) on Vercel + FastAPI backend + Qdrant Cloud/free tier.
  - Rationale: Aligns with best‑practice stack, supports modern UX, keeps infra cost low.
- **ADR‑002 — BGE‑M3 as Primary Embedding Model**
  - Decision: Use BGE‑M3 (local via Ollama/CPU) as default embedding model; Jina embeddings optional.
  - Rationale: Strong hybrid performance, no per‑call cost, good multilingual support.
- **ADR‑003 — LLM for Rerank/Summarise Only**
  - Decision: Use LLMs for reranking and answer generation, not every keystroke.
  - Rationale: Reduces cost and latency; keeps core search fast via Qdrant.
- **ADR‑004 — Public Repos Only in v1**
  - Decision: Restrict ingestion to public repos until security & ACL story is solid.
  - Rationale: Minimises data protection risk and OAuth complexity.

## Components
- **Ingestion Worker**
  - Responsibility: Collect/refresh GitHub repos, run summarisation, embeddings, and upsert to Qdrant.
  - Interfaces: CLI command (`ohmyrepos ingest`), internal Python APIs.
  - Dependencies: GitHub API, LLM providers, Qdrant.
- **Search Service (FastAPI)**
  - Responsibility: Serve search, RAG, and metadata endpoints.
  - Interfaces: HTTP/JSON; main entrypoint for UI and any future integrations.
  - Dependencies: Qdrant, LLM provider, configuration module, logging/monitoring.
- **UI (Next.js)**
  - Responsibility: Provide search UI, filters, and assistant.
  - Interfaces: Calls FastAPI endpoints; uses environment config for API base URL.
  - Dependencies: Vercel runtime, possibly Vercel edge functions for light tasks.
- **Core Libraries**
  - `src/core/*`, `src/llm/*`, `src/config/*` — shared logic between CLI, ingestion, and API.

## Code Standards & Conventions
### Language & Style
- Python 3.11+ with `ruff` (lint) and `black` (format) + `mypy` (type checking).
- TypeScript for Next.js with `eslint` + `prettier`.
- Consistent naming: `snake_case` for Python, `camelCase` for TS/JS, `PascalCase` for React components.
- Strict typing wherever practical; avoid `Any` except at boundaries.

### Framework & Project Layout
- Python:
  - `src/` as main package root; keep `core`, `llm`, `config`, `api` modules.
  - Separate FastAPI routers per functional area (`api/search.py`, `api/admin.py`, `api/health.py`).
- Next.js:
  - App Router structure with `app/(public)/search`, `app/(public)/repo/[id]`, etc.
  - Shared UI components in `components/` (search bar, filters, result cards).
- Environment configs:
  - `.env`, `.env.local` for local dev; secrets never committed.
  - Dedicated config for dev/stage/prod (flags in `src/config` and Next.js runtime config).

### API & Contracts
- JSON over HTTPS; explicit request/response schemas (Pydantic models for FastAPI).
- Clear error model: `{ "error": { "code": "...", "message": "...", "details": {...} } }`.
- Versioning:
  - Start with `/api/v1/...`; future breaking changes go to `/api/v2`.
- Validation:
  - Validate query length, filter values, page sizes; enforce sane limits.

### Testing
- Python:
  - `pytest` with coverage; focus on core retrieval, storage, and LLM wrappers.
  - Unit tests for embeddings, retriever, reranker.
  - Integration tests hitting Qdrant (with local/dev instance or mocked client).
- Next.js:
  - Component tests for critical UI parts (search input, results list).
  - Minimal E2E checks (e.g., Playwright/Cypress) for search flow.
- Targets:
  - Aim for 70%+ coverage on core modules; tests for all critical error paths.

### Security
- Auth:
  - For hosted version, GitHub OAuth for user login (minimum scopes).
  - No storage of access tokens beyond session lifetime; use refresh mechanisms cautiously.
- Secrets:
  - All API keys via env vars/secret store; never logged or stored in code.
- Dependencies:
  - Regular `pip-audit`/`pip-tools` or similar; lock files for Python and Node.
- Data protection:
  - No repo contents beyond what is public and necessary; respect GitHub ToS.
  - Protect against prompt injection and LLM misuse in “Ask the repos” feature.

### Resilience
- Timeouts on all network calls (GitHub, Qdrant, LLM providers).
- Retries with exponential backoff + jitter where idempotent.
- Circuit breakers for unstable providers (LLM APIs).
- Rate limiting at API boundary for high‑cost endpoints (LLM, heavy queries).

### Observability
- Structured JSON logs with correlation IDs (`request_id`).
- Metrics (even basic counters/timers) for:
  - Search requests, error rates, P95/P99 latencies.
  - LLM invocations and Qdrant queries.
- Health endpoints:
  - `/healthz` (lightweight) and `/readyz` (optionally checks Qdrant connectivity).

### Performance & Cost
- Budget:
  - Keep average search request under 500ms without LLM; under 1.5s with LLM rerank.
  - LLM spend target: <$5/month for personal use; guardrails on tokens/request.
- Optimisations:
  - Cache common queries or popular repo data (in‑memory or CDN).
  - Use batch queries where possible (Qdrant/LLM).

### Git & PR Process
- Branch naming: `feat/`, `fix/`, `chore/`, `docs/` prefixes with short slug.
- Conventional Commits style messages.
- PR checklist:
  - Tests updated/added.
  - Docs updated where relevant.
  - No secrets or debug artefacts.

### Tooling
- Python: `poetry` or `pip-tools` for dependency management (current `requirements.txt` can stay but should be reproducible).
- Pre‑commit hooks:
  - `ruff`, `black`, `mypy` for Python.
  - `eslint`, `prettier` for TS/JS.

### Commands (to be refined)
```bash
# Python
python -m pytest
ruff check src tests
black src tests
mypy src

# Next.js
pnpm lint
pnpm test
pnpm dev
```

### Anti-Patterns (Do NOT do this)
- No raw `print` debugging in production paths; use logger.
- No calls to LLMs without timeouts and budget limits.
- No hardcoded secrets, URLs, or magic constants.
- No direct DB/VectorDB calls from UI; always go through API.
- No unbounded result sets or unpaginated heavy endpoints.

### Configuration-Driven Policy
- All thresholds, limits, and external endpoints configured via env vars/config files.
- Validate config at startup; fail fast on missing/invalid required settings.
- Document key configuration options in README or dedicated config docs.

### File Creation Policy
- New files only for cohesive, reusable functionality.
- Avoid splitting by technical layer only; prefer feature‑oriented module boundaries where it helps DX.
- Remove obsolete Streamlit‑specific files once Next.js UI is live (after migration period).

## API Contracts (Draft)
- `GET /api/v1/healthz`
  - Returns basic service health.
- `POST /api/v1/search`
  - Input: `{ "query": string, "filters": {...}, "limit": int }`
  - Output: list of repos with scores, tags, and snippets.
- `POST /api/v1/ask`
  - Input: `{ "question": string, "repo_ids": [...optional...] }`
  - Output: answer text + cited repos/snippets.
- `GET /api/v1/repos/{id}`
  - Returns enriched metadata and stored summary for a repo.

## Data Model (High-Level)
- Qdrant collection `repos`:
  - Payload: `repo_id`, `name`, `owner`, `stars`, `language`, `topics`, `summary`, `tags`, `last_indexed_at`.
  - Vectors:
    - Dense vector(s) from BGE‑M3.
    - Optional sparse vector representation for BM25‑style search.
- Optional relational store (later):
  - `users`, `search_logs`, `collections` if/when multi‑user features are added.

## Quality & Operations
- Tests for ingestion and search must run fast and be deterministic.
- Observability:
  - Log every failed external call with context.
  - Track Qdrant/LLM errors separately from HTTP 5xx.
- Security checks as part of CI: dependency audit, secret scanning.

## Deployment & Platform Readiness
- FastAPI packaged as container; deploy to Render/Railway free tier.
- Next.js deployed to Vercel Hobby; configured with API base URL for backend.
- Resource considerations:
  - Respect memory/CPU limits of free tiers (avoid huge batch jobs in online path).
  - Long‑running ingestion jobs run locally or as short‑lived tasks, not in the main API dyno.

## Verification Strategy
- Pre‑deploy:
  - Run unit + integration tests locally/CI.
  - Smoke tests against dev Qdrant and GitHub.
- Post‑deploy:
  - Manual acceptance tests for key flows (search, ask, filters).
  - Monitor logs for errors and latency spikes.

## Affected Modules/Files
- Existing:
  - `src/core/*`, `src/llm/*`, `src/config/*`, `src/cli.py`, `src/app.py`.
- New/changed:
  - `src/api/` package for FastAPI routers.
  - Next.js app (separate repo or `/ui` folder).
  - Configs and deployment manifests (Dockerfiles, Render/Vercel configs).

## Implementation Steps (High-Level)
1. Extract and stabilise current ingestion/search core APIs in `src/core/*`.
2. Introduce FastAPI app with `/search`, `/ask`, `/healthz` endpoints wrapping core functions.
3. Build minimal Next.js UI consuming search API.
4. Wire observability (logging + minimal metrics) into ingestion and search paths.
5. Harden configuration, secrets management, and rate‑limit handling.
6. Iterate on UX (filters, result cards, “Ask the repos” assistant).

## Backlog (Tickets)
See `backlog/open/*.md` for concrete tickets:
- 01 — Platform & Config Setup
- 02 — FastAPI Search API
- 03 — Next.js UI MVP
- 04 — Ingestion Pipeline Refresh
- 05 — Observability & Healthchecks
- 06 — Cost & Quota Guardrails

## Interfaces & Contracts
- UI ↔ API: JSON over HTTPS; no direct DB/VectorDB access.
- API ↔ Qdrant: use official client with typed payloads.
- API ↔ LLMs: via provider abstraction; never call providers directly from business logic.

## Stop Rules & Preconditions
- Stop and re‑plan if:
  - Qdrant free tier cannot support required scale.
  - LLM costs exceed budget despite guardrails.
  - GitHub changes API/ToS in ways that affect ingestion.

## SLOs & Guardrails
- SLOs (non‑binding for free tier, but targets):
  - P95 search latency without LLM < 500ms.
  - P95 search latency with LLM < 1.5s.
  - Error rate < 2% on main search endpoint (excluding 4xx).
- Guardrails:
  - Max tokens per LLM call; max concurrent LLM calls per user/session.
  - Limits on page size and result count.

## Implementation Checklist
- [ ] API endpoints implemented and documented.
- [ ] Next.js UI deployed and wired to API.
- [ ] Ingestion pipeline migrated to new data model and Qdrant schema.
- [ ] Logging and minimal metrics in place.
- [ ] Secrets stored correctly; no secrets in repo.
- [ ] Core tests passing (Python + UI where applicable).

## Hidden Quality Loop (Internal)
- Periodically:
  - Review search quality using sample queries.
  - Re‑check cost posture (LLM and Qdrant usage).
  - Tighten tests and monitoring around the highest‑impact flows.

