### Tech stack, Domain, Year

* **Tech stack (target):**

  * Backend: Python 3.11+, FastAPI, `qdrant-client`, `httpx`
  * Retrieval: Qdrant Cloud (free tier), hybrid dense+sparse via Query API, BM25 handled by Qdrant
  * Embeddings: local **BGE-M3** (Ollama / FlagEmbedding) for default; optional **Jina embeddings v4** via API for advanced setups ([BGE Model][1])
  * LLMs: OpenRouter (multi-vendor, free tier + BYOK) + optional local Llama 3.1 / Mistral via Ollama ([OpenRouter][2])
  * Frontend: Next.js 14 (App Router) + React 18, deployed to Vercel Hobby (free) ([Northflank][3])
  * Hosting:

    * UI + thin edge API: Vercel (Hobby)
    * Python API worker: Render free web service or Railway free tier / $5 Hobby if needed ([Render][4])

* **Domain:** semantic search, tagging and RAG-style exploration over collections of GitHub repositories (starred/bookmarked, org catalogs, etc.).

* **Year:** 2025 (assume all limits/pricing are volatile → always re-check before going live; see TODOs in cost sections).

---

## 1. TL;DR (≤10 bullets)

1. **Architecture:** Single-page Next.js app + stateless FastAPI backend + Qdrant Cloud free cluster (1 GB / ~1M 768-dim vectors) with hybrid search (dense + sparse) replaces home-rolled BM25 infra. ([Qdrant][5])
2. **Retrieval core:** Use **BGE-M3** as the default embedding model (local via Ollama or CPU server) – one model gives you dense + sparse + multi-vector behavior, ideal for hybrid search over code & text. ([BGE Model][1])
3. **LLM use:** Keep LLM in the **rerank + summarise** loop, not in every keystroke. Use OpenRouter free/cheap models for reranking & answer generation with hard token ceilings; reserve local models for offline mode. ([OpenRouter][6])
4. **Cost posture:**

   * Indexing: run batch jobs locally; only push embeddings to Qdrant Cloud free tier.
   * Online: aim for **< $5/month** LLM+embedding spend; default to local embeddings and only call remote LLMs for “view details / ask question” flows.
5. **Performance SLOs (MVP):**

   * P95 API search latency (no LLM): **<500 ms**, end-to-end with LLM rerank: **<1.5 s**, P99 <3 s.
   * Availability: **99%** for API/UI on free tiers (best-effort; allow cold-start lag on Render/Vercel). ([Render][4])
6. **UX posture:** Modern Next.js UI with: instant lexical search, “smart” semantic suggestions, filters (language, stars, activity), and a separate “Ask the repos” assistant pane wired to the RAG pipeline.
7. **Quality posture:** Hybrid retrieval + reranking as default: BM25 for exact matches; dense for semantics; optional Jina / jina-reranker-v3 for deep rerank when needed. ([Qdrant][7])
8. **Observability:** Minimal but concrete: structured JSON logs, a few key metrics (latency, error rate, LLM calls, Qdrant query counts), plus cheap uptime checks and log-based alerts.
9. **Security posture:** Follow **OWASP Top 10:2025** (web/API) + **OWASP Top 10 for LLM Apps 2025**; GitHub OAuth only, least-privilege scopes, no long-lived PATs; no private repo ingestion in v1. ([OWASP Foundation][8])
10. **Roadmap:** Pattern A (MVP, mostly free tier). Pattern B (scale-up: bigger Qdrant cluster, dedicated Postgres, paid LLMs, proper observability stack) with clean migration path.

---

## 2. Landscape — What’s new in 2025

### 2.1 Retrieval, embeddings, and RAG

* **Hybrid search is “table stakes”**
  Modern RAG/retrieval guides strongly recommend **hybrid retrieval** (BM25 + dense) as default; pure vector or pure lexical is now considered a deliberate trade-off, not a default. ([Morphik][9])

* **Qdrant’s Query API & hybrid queries**
  Qdrant 1.10+ adds a **Query API** and **hybrid/multi-stage queries** so you can combine sparse and dense vectors on the server without additional infra. ([Qdrant][7])
  For ohmyrepos this means you can retire a separate BM25 engine and keep BM25-style search inside Qdrant (sparse vectors or text index), simplifying the stack.

* **Embedding model landscape (2025):**

  * **BGE-M3** — multi-lingual, multi-granularity, supports dense, multi-vector and sparse retrieval, up to 8,192 tokens, targeted explicitly at hybrid search use-cases. ([BGE Model][1])
  * **Jina embeddings v4** — multi-modal, multilingual, supports late-interaction and multi-vector outputs, designed to unify text + images (nice if you later ingest repo screenshots, READMEs with diagrams, etc.). ([Jina AI][10])
  * Multiple independent leaderboards show open-source models (BGE, Jina) approaching or matching closed options for many RAG settings, especially multilingual search. ([Agentset][11])

* **Rerankers**

  * **jina-reranker-v3** (2025) offers strong multilingual reranking with a “last but not late” interaction architecture. ([Jina AI][10])
  * Several 2024–2025 blogs benchmark rerankers as a **cheap, high-impact layer** for RAG quality. ([Analytics Vidhya][12])

* **RAG evaluation**
  2024–2025 brought better guidance and even survey papers for RAG evaluation: retrieval metrics, answer-level metrics, safety, etc. ([Qdrant][13])
  For ohmyrepos, that translates into small test sets of “gold” repo matches per query and a basic automated eval job.

### 2.2 Hosting & cost

* **Qdrant Cloud**: free 1GB cluster, ~1M vectors (768-dim) with no credit card, ideal for ohmyrepos scale and budget. ([Qdrant][5])
* **Vercel Hobby (2025):** free plan targeted at **personal/non-commercial** use, ~100 GB bandwidth/month and 100k serverless invocations/month; good enough for early ohmyrepos. ([Northflank][3])
* **Render free:** free web service & Postgres/Key-Value with 750 free instance-hours; services spin down after inactivity → cold-starts up to ~1 minute. Good for a small FastAPI backend if you accept cold starts. ([Render][4])
* **Railway:** now more of a freemium/paid platform – 30-day free trial with $5 credit and a Free plan with 0.5GB RAM / 1 vCPU; realistically $5 Hobby for always-on. ([Railway][14])

### 2.3 LLM access

* **OpenRouter**

  * Unified API for 300–500+ models with **free tier**, plus BYOK with 1M free BYOK requests/month (as of Oct 2025). ([OpenRouter][6])
  * Multiple providers have their own free credits (Mistral, Google, Meta etc.), and blog posts explicitly target “use LLM APIs for free in 2025” as a pattern. ([Teamday][15])

* **Open-source LLMs**

  * Llama 3.1 / 4 families, Mistral/Mixtral and derivatives give good local or self-hosted options for summarisation/rerank or full chat. ([Meta AI][16])

### 2.4 Security

* **OWASP Top 10:2025 (web/API)** final/RC available – shifts emphasis to **software supply chain** and **exception handling** as first-class risk categories. ([OWASP Foundation][8])
* **OWASP Top 10 for LLM Apps (2025)**: adds LLM-specific categories such as **Prompt Injection**, **Sensitive Info Disclosure**, **Data Poisoning**, **Improper Output Handling** and **Excessive Agency**. ([Confident AI][17])

---

## 3. Architecture Patterns (ohmyrepos with FastAPI + Next.js)

### Pattern A — “Free-tier Cloud MVP”

**When to use**

* You want a shareable hosted version, minimal infra, minimal cost.
* One maintainer, low traffic, mostly personal use or light community use (< few K searches/day).

**Components**

* **Frontend:** Next.js 14 on Vercel Hobby.
* **Backend API:** FastAPI on Render Free web service.
* **Vector store:** Qdrant Cloud free cluster.
* **LLM/embeddings:**

  * Embeddings: local BGE-M3 during indexing (run from laptop/cheap VPS) → store vectors in Qdrant. ([Ollama][18])
  * Online LLM: OpenRouter free tier for reranking or answer generation; optional local model via Ollama for dev/offline.

**High-level flow**

1. **Ingestion job** (local or cheap VPS)

   * Pull GitHub metadata (stars, languages, topics, README, maybe top files).
   * Chunk text + code; compute dense & sparse embeddings with BGE-M3; upsert to Qdrant with metadata tags (repo, path, language, stars, updated_at). ([BGE Model][1])
2. **Search API**

   * Receive query + filters.
   * Run **hybrid Qdrant query** (dense + sparse, optional BM25-like weights). ([Qdrant][19])
   * Optionally pass top-k results through local or OpenRouter LLM/reranker for ranking and summarisation. ([Jina AI][20])
3. **UI**

   * Search bar + filters; show results list.
   * “Ask the repos” opens side panel: call RAG chain (retrieve→rerank→build context→call LLM→display answer with sources).

**Pros**

* Almost all infra on **free tiers**; Qdrant cluster fits ohmyrepos scale. ([Qdrant][5])
* Clear separation between UI (Vercel) and API (Render) – easier to reason about limits.
* BGE-M3 + Qdrant hybrid search is a strong baseline for code/search tasks. ([Zilliz][21])

**Cons**

* Render free dynos sleep → cold start; heavy indexing jobs must run elsewhere. ([Render][22])
* Free tiers are not contractual; quotas & terms may change (must periodically re-check).
* LLM calls via OpenRouter free tier are **rate-limited** and best-effort; can’t guarantee latency or availability. ([OpenRouter][6])

**Optional later features**

* Add per-user “collections” and saved searches.
* Add GitHub OAuth and limited private-repo search (requires stricter security review).
* Add tagging/auto-labeling pipelines using small local LLMs.

---

### Pattern B — “Scale-up / Team Mode”

**When to use**

* Multi-user teams, heavier usage, or you want reliable uptime for demoing/consulting.
* You are fine with small monthly infra spend (say $20–50).

**Changes vs Pattern A**

* **Backend** moves to paid instance:

  * Render paid web service, or Railway Hobby + reserved compute. ([GetDeploying][23])
* **Vector store**:

  * Upgrade Qdrant Cloud to paid plan if >1M vectors or higher throughput needed. ([Qdrant][24])
* **DB**: add Postgres (Render free/paid) for users, sessions, and job tracking. ([Render][4])
* **LLM/embeddings**:

  * Move to pay-as-you-go LLMs and embeddings (OpenRouter pay-as-you-go, OpenAI, Voyage, etc.), possibly with BYOK for vendor flexibility. ([OpenRouter][6])

**Migration from A**

* Use the same Qdrant collections and metadata schema; upgrading Qdrant Cloud is configuration, not code. ([Qdrant][5])
* Move backend deployment from free Render to a paid instance; keep same FastAPI container image.
* Introduce feature flags for “team-only” features (multi-user, org collections, advanced RAG).

---

## 4. Priority 1 — Retrieval & Indexing (Hybrid Search over GitHub Repos)

### Why

* Directly tied to the core goal: **fast, accurate search** over many repos.
* Mitigates key risks: poor search quality, high infra cost, inability to scale from “a few repos” to “hundreds+”.

### Scope

* **In scope:**

  * Chunking & metadata schema for repos.
  * Embedding model choice & pipeline.
  * Qdrant schema, hybrid search queries, index maintenance.
* **Out of scope (for P1):**

  * Fancy RAG “agent” flows.
  * Per-user private repositories and complex ACLs.

### Key decisions (with rationale)

1. **Use BGE-M3 as primary embedding model; Qdrant for dense + sparse**

   * Rationale: BGE-M3 supports dense + sparse + multi-vector retrieval and >100 languages, ideal for hybrid search without juggling multiple models. ([BGE Model][1])
   * Alternative: `text-embedding-3-large` (OpenAI), Jina embeddings v4; better absolute quality but adds external API cost/dependency. ([ZenML][25])

2. **Qdrant Cloud free tier for MVP**

   * Rationale: 1GB/~1M vectors, hybrid search, managed cluster – perfect for ohmyrepos scale, free. ([Qdrant][5])
   * Alternative: Postgres + pgvector, Meilisearch, Elasticsearch; but then you lose easy hybrid vector + sparse integration in one service. ([Tiger Data][26])

3. **Hybrid retrieval default: lexical + dense**

   * Rationale: research & practitioner reports consistently show improved RAG accuracy when combining BM25-like and dense embeddings. ([Morphik][9])

4. **Minimal schema with stable, queryable metadata**

   * Rationale: filters (language, stars, last updated) are critical for developer workflows; Qdrant’s filtering and hybrid query docs emphasise structuring metadata for efficient filtering. ([Qdrant][27])

### Implementation outline (3–6 concrete steps)

Below: a sketched P1 plan you can implement in a weekend.

#### Step 1 — Design Qdrant schema

One collection, e.g. `repos`, with:

* **Vectors:**

  * `dense`: 1024-dim BGE-M3 dense embedding.
  * `sparse`: sparse vector for BM25-like scoring (via BGE-M3 sparse output or Qdrant text index). ([Qdrant][19])
* **Payload fields (metadata):**

  * `repo_full_name` (string, e.g. "owner/name")
  * `path` (string, e.g. "README.md", "src/api/client.py")
  * `kind` ("readme" | "code" | "issue" | "changelog" | ...)
  * `language` (string)
  * `stars` (int)
  * `last_pushed_at` (timestamp or date)
  * `size` (int, bytes or LoC)
  * `url` (string, GitHub URL)

You can create this via `qdrant-client`:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

client.recreate_collection(
    collection_name="repos",
    vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams(index=True)},
)
```

#### Step 2 — Chunking & embedding pipeline

* **Chunking strategy (text):**

  * For README/docs: sliding window, 300–500 tokens with 50–100 token overlap; keep sections (headers) as metadata.
  * For code: function/method/class-level chunks where possible; fall back to file-level slices if parsing fails.
    This matches RAG/document-processing best practices that emphasize chunking by semantics + structure rather than naive fixed length. ([LinkedIn][28])

* **Embedding:**

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def encode_chunk(text: str):
    out = model.encode(
        [text],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"][0]
    sparse = out["sparse_vecs"][0]
    return dense, sparse
```

* **Upsert to Qdrant:**

```python
from qdrant_client.models import PointStruct

points = []
for i, chunk in enumerate(chunks):
    dense, sparse = encode_chunk(chunk.text)
    points.append(
        PointStruct(
            id=chunk.id,
            vector={"dense": dense, "sparse": sparse},
            payload={
                "repo_full_name": chunk.repo_full_name,
                "path": chunk.path,
                "kind": chunk.kind,
                "language": chunk.language,
                "stars": chunk.stars,
                "last_pushed_at": chunk.last_pushed_at.isoformat(),
                "url": chunk.url,
                "text": chunk.text[:2000],  # preview snippet
            },
        )
    )

client.upsert("repos", points=points, wait=True)
```

#### Step 3 — Hybrid search query

Use Qdrant’s **Query API** with both dense and sparse vectors for a query. ([Qdrant][19])

```python
def search(q: str, filters: dict | None = None, limit: int = 20):
    enc = model.encode(
        [q],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = enc["dense_vecs"][0]
    sparse = enc["sparse_vecs"][0]

    must_filters = []
    # build `must_filters` from filters, e.g. language==python, stars>50

    return client.query_points(
        collection_name="repos",
        query={
            "must": must_filters,
        },
        query_filter=None,
        # Hybrid query: dense + sparse
        query_vector=[
            {"name": "dense", "vector": dense, "weight": 0.7},
            {"name": "sparse", "vector": sparse, "weight": 0.3},
        ],
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
```

Weights (0.7/0.3) are tunable; consider making them a per-query knob or heuristics (e.g. if query has many symbols, rely more on sparse).

#### Step 4 — Reranking (optional but recommended)

* For top-k (say 50) results, call a **reranker** (local or Jina reranker via API) to improve ordering. ([Jina AI][20])

Pseudocode:

```python
def rerank(query: str, results):
    # results: list of (id, payload)
    pairs = [(query, r.payload["text"]) for r in results]
    scores = call_reranker_api(pairs)  # or local model
    return [r for _, r in sorted(zip(scores, results), reverse=True)]
```

Guard this behind a feature flag and only enable for “slow mode” or “Ask the repos” to keep costs low.

### Guardrails & SLOs

* **Latency:**

  * Retrieval-only P95 <500 ms; RAG end-to-end P95 <1.5s.
* **Quality:**

  * On a small eval set (e.g. 50–100 queries with ground truth repos/files), hit@5 ≥ 0.8 (80% of the time, a correct result is in top-5).
* **Cost:**

  * Indexing: run batch job locally; Qdrant free tier only.
  * Online: reranker + LLM per query only on “detailed view” or “Ask” flows → keep daily LLM calls < 500 on free tiers.
* **TODO (before launch):**

  * Re-check Qdrant free tier limits & pricing. ([Qdrant][5])

### Failure modes & recovery

* **Qdrant throttling / errors**

  * Detection: elevated 5xx from Qdrant, timeouts; log `qdrant_latency_ms` and error counts.
  * Remediation: exponential backoff, degrade to GitHub search link fallback, or local BM25 fallback (for local usage).
* **Embedding model regression**

  * Detection: nightly/weekly RAG eval job; track hit@k. ([Qdrant][13])
  * Remediation: keep previous embedding model around; support “blue/green” collections and switch back.

---

## 5. Priority 2 — Hosting & Deployment (Free-tier Cloud)

### Why

* Hosting is where **“free tier illusions”** kill projects: CPU hours, vector DB usage, LLM APIs all have soft/hard caps.
* Good default architecture avoids re-platforming for months.

### Scope

* In scope: UI, API, Qdrant Cloud, CI/CD basics.
* Out of scope: multi-region, on-prem, high-availability setups.

### Decisions

1. **Vercel + Render combo for MVP**

   * Rationale: Vercel is excellent at static/Next.js hosting on the free Hobby plan; Render free web service handles long-running Python API & background jobs. ([Northflank][3])

2. **Qdrant Cloud, not self-hosted**

   * Rationale: Managed, 1GB free cluster, capacity for ~1M vectors, plus hybrid search and observability built-in; avoids running your own DB. ([Qdrant][5])

3. **CI/CD via GitHub → Vercel & Render**

   * Rationale: both have native GitHub integrations; simple to set up.

### Implementation outline

1. **Backend (FastAPI) on Render**

* Dockerfile or pure build:

```yaml
# render.yaml
services:
  - type: web
    name: ohmyrepos-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    plan: free  # later upgrade
```

* Env vars: `QDRANT_URL`, `QDRANT_API_KEY`, `OPENROUTER_API_KEY`, etc.

2. **Frontend (Next.js) on Vercel**

* Connect GitHub repo, import project.
* Set env vars: `NEXT_PUBLIC_API_BASE_URL`, etc.
* Configure caching:

  * Statically render marketing/docs pages.
  * Use client-side fetch to backend search API (avoid SSR hitting free limits too hard). ([Northflank][3])

3. **Qdrant Cloud**

* Create free cluster, note endpoint and API key. ([cloud.qdrant.io][29])
* Run a local ingestion script that connects to Qdrant Cloud.

### Guardrails & SLOs

* **Cold start tolerance:**

  * Accept 10–60 s cold start on Render free instances; show “Warming up” UI if API takes >3 s and retry. ([Render][22])
* **Quota monitoring (TODO):**

  * Add monthly check or script to read Vercel/Render/Qdrant usage dashboards; raise alerts when approaching 80% of free tier. ([Northflank][3])

---

## 6. Priority 3 — LLM Usage & Cost Optimisation

### Why

* LLM calls are the easiest way to blow up costs and latency.
* For ohmyrepos, **search** is the main value; LLM is “icing”.

### Scope

* In scope:

  * Rerank, summarisation, Q&A from retrieved context.
  * Provider strategy (OpenRouter + local).
* Out of scope:

  * Multi-agent workflows, complex tools execution.

### Decisions

1. **“Retrieval-first” UX**

   * Every query always shows **raw search results** first; LLM answer is **explicit opt-in** (“Explain top results”, “Ask the repos”).

2. **OpenRouter as primary gateway; local models as fallback**

   * Rationale: free tier + BYOK; easy switching between models; widely documented for 2025. ([OpenRouter][6])

3. **Token budgets per request**

* For Q&A:

  * Context: ≤3–4 chunks, each ≤512 tokens → ≤2k tokens.
  * Prompt + answer: target ≤1k–1.5k tokens.
* Hard limit: 4096 tokens per request to avoid surprise bills.

### Implementation outline

* **Rerank**: prefer a small reranker (e.g., jina-reranker-v3) instead of a full chat model when only reordering results. ([Hugging Face][30])
* **Q&A**:

```python
def answer_query(query: str):
    hits = search(query, limit=20)
    hits = rerank(query, hits)[:4]

    context = "\n\n".join(
        f"[{i+1}] {h.payload['repo_full_name']}:{h.payload['path']}\n{h.payload['text']}"
        for i, h in enumerate(hits)
    )

    prompt = f"""You are an assistant helping developers discover and compare GitHub repositories.

Question: {query}

Context snippets:
{context}

Answer briefly (<= 5 bullet points) and reference snippets by [index]."""

    return call_openrouter_chat(model="meta-llama/llama-3.1-8b-instruct", prompt=prompt)
```

* **Rate limiting:**

  * Client-side: disable “Ask” button while a request is in flight.
  * Server-side: simple per-IP or per-user limit (e.g. 60 Q&A calls/day).

### Guardrails & SLOs

* **Cost:**

  * Keep monthly Q&A token usage ≤ 1–2M tokens during early phase (under a few dollars on most providers). ([Teamday][15])
  * Prefer free models / free credits initially; move to paid when stable. ([Cursor IDE中文站][31])

* **TODO (before launch):**

  * Re-check OpenRouter free tier limits and BYOK policies. ([OpenRouter][6])

---

## 7. Testing Strategy (FastAPI + Qdrant + RAG)

### Levels

1. **Unit tests (Python & JS)**

   * Python: test chunking, embedding wrapper, Qdrant query builder, scoring logic.
   * JS: test small UI components and data mappers.

2. **Integration tests**

* Use a small **test Qdrant collection** with a dozen repos and known queries.
* Validate retrieval: queries like “fastapi auth middleware example” must return the expected repos in top-k.

3. **E2E tests**

* Use Playwright or Cypress to script browser flows: type query, see results, click “Ask the repos”, verify answer contains expected repo names.

4. **Performance tests**

* For retrieval: simple Python script that runs 100 representative queries and records latency distribution.
* For RAG: subset of queries with Q&A; measure end-to-end.

5. **Security tests**

* Lint URLs; test that only GitHub OAuth flows exist; no secrets leak into logs.
* For LLM, add prompt-injection “canary” tests using OWASP examples. ([OWASP Gen AI Security Project][32])

---

## 8. Observability & Operations

### Metrics

* **Backend API:**

  * `http_requests_total` (by route, status).
  * `http_request_duration_ms` (p50/p90/p95).
  * `llm_calls_total` (by provider/model).
  * `qdrant_query_duration_ms`.

* **Retrieval quality:**

  * `rag_eval_hit_at_5` from nightly/weekly job (small dataset). ([Qdrant][13])

### Logging

* JSON logs with request ID; include Qdrant latency, LLM latency, truncated prompts (without repo content beyond what’s necessary for debugging).
* Avoid logging full GitHub access tokens or OAuth details.

### Alerting (minimal)

* Uptime check for API and main UI path (GitHub Actions + ping service or free uptime monitor).
* Alert when:

  * Error rate > 5% over 5 minutes.
  * P95 latency > 2 s over 5–10 minutes.

---

## 9. Security Best Practices

### Web/API (OWASP Top 10:2025)

* **Broken Access Control & Security Misconfiguration**

  * Simple: for MVP, no user-specific private data; everything read-only from public GitHub; no admin panel.
  * Lock down CORS to your domain; no wildcard `*`. ([OWASP Foundation][8])

* **Supply Chain Failures**

  * Pin dependencies; use Dependabot / Renovate; run `pip-audit` & `npm audit` regularly. ([OWASP Foundation][8])

### LLM-specific (OWASP Top 10 for LLM Apps 2025)

* **LLM01 Prompt Injection / LLM05 Improper Output Handling**

  * Never execute code from LLM suggestions automatically.
  * When showing repo commands, treat them as plain text; do not embed any auto-run controls. ([Confident AI][17])

* **LLM02 Sensitive Data Disclosure**

  * MVP: only public GitHub data; do not ingest secrets or private repos.
  * If adding private repos later, implement per-user indexes or filters + strict scoping. ([OWASP Foundation][33])

### Data protection & secrets

* Store secrets (Qdrant key, OpenRouter key) only in hosting env vars.
* Rotate keys periodically; never log them.

---

## 10. Performance & Cost

### Budgets (concrete but revisitable)

* **Vector DB:** stay within Qdrant free cluster

  * ≤1M vectors, 768–1024 dims, 4GB disk. ([Qdrant][5])
* **Hosting:** stay within Vercel & Render free tiers for MVP. ([Northflank][3])
* **LLM & embeddings:**

  * Prefer local embeddings; limit remote LLM tokens to ≤ 2M/month initially. ([Elephas][34])

### Optimisation techniques

* Aggressive caching of search results & RAG answers in memory/Key-Value store (e.g., Render key-value). ([Render][4])
* Use `top_k`=20 or so for retrieval; only send ≤4–6 chunks to LLM.
* Precompute **repo-level embeddings** for “show similar repos” to avoid heavy online queries.

### TODO checks

* Re-verify Vercel Hobby limits (bandwidth, function invocations) and Render free instance limits before promoting to public beta. ([Northflank][3])

---

## 11. CI/CD Pipeline

Minimum pipeline (GitHub Actions):

1. **On PR:**

   * Lint (Python, JS/TS).
   * Unit tests.
   * Small integration test suite hitting a disposable Qdrant collection.

2. **On push to main:**

   * Same as PR.
   * Deploy frontend to Vercel (automatic).
   * Deploy backend to Render using Render’s GitHub integration or CLI.

Example GitHub Actions steps: (outline only)

* `backend-ci.yml`:

  * Setup Python, install deps, run `pytest`.
* `frontend-ci.yml`:

  * Setup Node, install, run `npm test` / `npm run lint`.

---

## 12. Code Quality Standards

* **Python:**

  * PEP8 via `ruff` or `flake8`; autoformat with `black`; type-check with `mypy`.
  * Use Pydantic v2 models for request/response schemas.

* **TypeScript/JS:**

  * ESLint + Prettier; strict TS mode; React testing library for key components.

* **Review & refactoring:**

  * No direct calls to LLM / Qdrant from components; everything through small service modules for easier mocking and replacement.

---

## 13. Reading List (with dates & gists)

1. **Qdrant – Hybrid Search Revamped (2024-07-25)** — how to use Query API for hybrid dense+sparse search within Qdrant; includes patterns and examples. ([Qdrant][7])
2. **Qdrant Cloud Docs & Pricing (2024–2025)** — explains free 1GB cluster and capacity estimates (~1M 768-dim vectors). ([Qdrant][5])
3. **BGE-M3 paper & docs (2024–2025)** — multi-functional, multilingual embedding model for dense + sparse + multi-vector retrieval. ([arXiv][35])
4. **Jina embeddings v4 paper (2025-07-07)** — multimodal embedding model with late-interaction and strong retrieval performance. ([arXiv][36])
5. **RAG Evaluation Guides (2024–2025)** — Qdrant, Meilisearch, Orq.ai posts; plus 2025 survey paper on RAG evaluation metrics and frameworks. ([Qdrant][13])
6. **OpenRouter Pricing & Free Tier (2025)** — docs and blog posts on free usage, BYOK, and provider free credits. ([OpenRouter][6])
7. **OWASP Top 10:2025 & LLM Top 10 (2024–2025)** — core security risk lists for web apps and LLM apps. ([OWASP Foundation][8])
8. **“Best Embedding Models 2025” & open-source benchmarks (2025)** — comparative guides on OpenAI vs BGE vs Jina vs Voyage for RAG. ([Elephas][34])
9. **Document Processing for RAG (2025-11-02)** — practical advice on chunking, metadata, hybrid search, and containerized deployments. ([Collabnix][37])

---

## 14. Decision Log (ADR-style, condensed)

* **ADR-001 – Qdrant Cloud over self-hosted vector DB**

  * Chosen because: free managed 1GB cluster, built-in hybrid search, good docs; avoids ops toil.
  * Alternatives: pgvector, Pinecone, Elasticsearch. ([Qdrant][5])

* **ADR-002 – BGE-M3 as main embedding model**

  * Chosen because: multi-lingual, dense+sparse support, strong RAG benchmarks, good local story.
  * Alternatives: OpenAI `text-embedding-3-large`, Jina embeddings v4, Voyage. ([BGE Model][1])

* **ADR-003 – Vercel + Render for free-tier hosting**

  * Chosen because: simple integration, both have workable free tiers and strong Next.js / Python support.
  * Alternatives: Netlify, Fly.io, Railway only, single-platform hosting. ([Northflank][3])

* **ADR-004 – Retrieval-first UX with optional LLM layer**

  * Chosen because: preserves utility even if LLM quota exhausted; easier to reason about cost and latency.
  * Alternatives: chat-first UI where every keystroke hits the LLM.

---

## 15. Anti-Patterns to Avoid

1. **Monolithic “RAG for everything”**

   * Problem: mixing retrieval, ranking, summarisation, and action in a single giant prompt makes debugging and cost control hard.
   * Instead: keep **retrieval → (optional rerank) → context assembly → LLM** as distinct, small functions. ([Qdrant][13])

2. **Indexing entire repos as one giant document**

   * Problem: terrible chunk recall, high token waste; fails all RAG best-practice docs.
   * Instead: chunk by file/function/section, with meaningful metadata. ([LinkedIn][28])

3. **Always calling LLM for search**

   * Problem: costs explode; latency unpredictable; search becomes non-deterministic.
   * Instead: search defaults to pure hybrid retrieval; LLM only when explicitly requested.

4. **Ignoring free-tier quirks**

   * Problem: Render free services spinning down; Vercel Hobby limits; Qdrant free cluster size.
   * Instead: design for cold starts, set budgets, and re-check quota docs regularly. ([Render][22])

5. **Skipping security basics because “it’s just a toy”**

   * Problem: even toy apps can leak tokens or be used for phishing.
   * Instead: follow OWASP Top 10 & LLM Top 10 baselines even in MVP. ([OWASP Foundation][8])

---

## 16. Evidence & Citations

* The guide above references:

  * Qdrant docs/blogs on hybrid search, Query API, and free tier. ([Qdrant][5])
  * Research/blogs on RAG evaluation and hybrid retrieval. ([Qdrant][13])
  * Documentation and papers for BGE-M3 and Jina embeddings v4. ([BGE Model][1])
  * 2025 resources on free-tier hosting and OpenRouter free/cheap usage. ([Northflank][3])
  * 2025 OWASP Top 10 (web + LLM) and AI Security/Privacy guides. ([OWASP Foundation][8])

Where pricing/limits are mentioned, treat them as **indicative only** and re-confirm them from the primary docs before hard-wiring them into ohmyrepos.

---

## 17. Verification

### Self-check for ohmyrepos

1. **Retrieval**

   * Build a small eval set (50–100 queries with labeled repos/files).
   * Scripted RAG eval: measure hit@5, hit@10; compare pure dense vs sparse vs hybrid. ([Qdrant][13])

2. **Performance**

* Run a load script with 100–200 queries from a laptop against the deployed API.
* Ensure P95 <1.5s with LLM off; with LLM on only for “Ask” flows.

3. **Cost**

* One month after launch, export billing/usage from:

  * Qdrant Cloud, Vercel, Render, OpenRouter.
* Validate you are within budgets; if not, adjust caching and rate limits.

### Confidence levels

* Retrieval architecture & hybrid search: **High** (strong convergence in research and vendor docs).
* Exact model choices (BGE-M3 vs alternatives): **Medium** (depends on your language mix and code/document types; benchmarks differ).
* Free-tier cost planning: **Medium-Low** (providers frequently adjust quotas & terms; always re-check).

[1]: https://bge-model.com/bge/bge_m3.html?utm_source=chatgpt.com "BGE-M3 — BGE documentation"
[2]: https://openrouter.ai/?utm_source=chatgpt.com "OpenRouter"
[3]: https://northflank.com/blog/vercel-vs-netlify-choosing-the-deployment-platform-in-2025?utm_source=chatgpt.com "Vercel vs Netlify: Choosing the right one in 2025 (and what ..."
[4]: https://render.com/docs/free?utm_source=chatgpt.com "Deploy for Free"
[5]: https://qdrant.tech/documentation/cloud/create-cluster/?utm_source=chatgpt.com "Creating a Qdrant Cloud Cluster"
[6]: https://openrouter.ai/pricing?utm_source=chatgpt.com "Pricing"
[7]: https://qdrant.tech/articles/hybrid-search/?utm_source=chatgpt.com "Hybrid Search Revamped - Building with Qdrant's Query API"
[8]: https://owasp.org/Top10/2025/0x00_2025-Introduction/?utm_source=chatgpt.com "Introduction - OWASP Top 10:2025 RC1"
[9]: https://www.morphik.ai/blog/retrieval-augmented-generation-strategies?utm_source=chatgpt.com "RAG in 2025: 7 Proven Strategies to Deploy Retrieval- ..."
[10]: https://jina.ai/?utm_source=chatgpt.com "Jina AI - Your Search Foundation, Supercharged."
[11]: https://agentset.ai/embeddings?utm_source=chatgpt.com "Embedding Model Leaderboard"
[12]: https://www.analyticsvidhya.com/blog/2025/06/top-rerankers-for-rag/?utm_source=chatgpt.com "Top 7 Rerankers for RAG"
[13]: https://qdrant.tech/blog/rag-evaluation-guide/?utm_source=chatgpt.com "Best Practices in RAG Evaluation: A Comprehensive Guide"
[14]: https://railway.com/pricing?utm_source=chatgpt.com "Railway Pricing and Plans"
[15]: https://www.teamday.ai/blog/top-ai-models-openrouter-2025?utm_source=chatgpt.com "Top AI Models on OpenRouter 2025: Cost vs Performance ..."
[16]: https://ai.meta.com/blog/meta-llama-3-1/?utm_source=chatgpt.com "Introducing Llama 3.1: Our most capable models to date"
[17]: https://www.confident-ai.com/blog/owasp-top-10-2025-for-llm-applications-risks-and-mitigation-techniques?utm_source=chatgpt.com "OWASP Top 10 2025 for LLM Applications"
[18]: https://ollama.com/library/bge-m3?utm_source=chatgpt.com "bge-m3"
[19]: https://qdrant.tech/documentation/concepts/hybrid-queries/?utm_source=chatgpt.com "Hybrid Queries"
[20]: https://jina.ai/en-US/reranker/?utm_source=chatgpt.com "Reranker API"
[21]: https://zilliz.com/ai-models/bge-m3?utm_source=chatgpt.com "The guide to bge-m3 | BAAI"
[22]: https://render.com/docs/faq?utm_source=chatgpt.com "Render FAQ – Render Docs"
[23]: https://getdeploying.com/railway?utm_source=chatgpt.com "Railway | Review, Pricing & Alternatives"
[24]: https://qdrant.tech/pricing/?utm_source=chatgpt.com "Pricing for Cloud and Vector Database Solutions Qdrant"
[25]: https://www.zenml.io/blog/best-embedding-models-for-rag?utm_source=chatgpt.com "9 Best Embedding Models for RAG to Try This Year"
[26]: https://www.tigerdata.com/blog/pgvector-vs-qdrant?utm_source=chatgpt.com "Pgvector vs. Qdrant: Open-Source Vector Database ..."
[27]: https://qdrant.tech/articles/vector-search-manuals/?utm_source=chatgpt.com "Vector Search Manuals"
[28]: https://www.linkedin.com/posts/maxirwin_hybrid-search-revamped-qdrant-activity-7032428696170876928-Uc9o?utm_source=chatgpt.com "Max Irwin - Building with Qdrant's Query API"
[29]: https://cloud.qdrant.io/?utm_source=chatgpt.com "Qdrant Cloud"
[30]: https://huggingface.co/jinaai/jina-reranker-v3?utm_source=chatgpt.com "jinaai/jina-reranker-v3"
[31]: https://www.cursor-ide.com/blog/free-claude-openrouter-guide-2025?utm_source=chatgpt.com "Free Claude Models via OpenRouter: Complete Guide 2025"
[32]: https://genai.owasp.org/resource/llm-top-10-for-llms-v1-1/?utm_source=chatgpt.com "LLM Top 10 for LLMs 2024 - OWASP Gen AI Security Project"
[33]: https://owasp.org/www-project-top-10-for-large-language-model-applications/?utm_source=chatgpt.com "OWASP Top 10 for Large Language Model Applications"
[34]: https://elephas.app/blog/best-embedding-models?utm_source=chatgpt.com "13 Best Embedding Models in 2025: OpenAI vs Voyage AI ..."
[35]: https://arxiv.org/abs/2402.03216?utm_source=chatgpt.com "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"
[36]: https://arxiv.org/pdf/2506.18902?utm_source=chatgpt.com "jina-embeddings-v4"
[37]: https://collabnix.com/document-processing-for-rag-best-practices-and-tools-for-2024/?utm_source=chatgpt.com "Document Processing for RAG: Best Practices and Tools ..."
