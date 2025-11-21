# Ticket 06: Cost & Quota Guardrails

**Goal**: Implement mechanisms to prevent cost overruns and abuse of the free tier resources.

## Context
We are running on free tiers (Qdrant Cloud, OpenRouter free/cheap models). We must ensure we don't hit hard limits or incur unexpected costs.

## Requirements
- [ ] **Token Counting**:
    - Implement a utility to estimate tokens for a given text (using `tiktoken` or character count heuristic).
    - Log token usage for every LLM call.
- [ ] **Budget Enforcement**:
    - Add a simple in-memory or file-based counter for daily token usage.
    - If usage > limit (e.g., 1M tokens/month prorated), reject requests with 429.
    - *Note*: For MVP, a simple env var `MAX_DAILY_TOKENS` and in-memory counter (reset on restart is fine for now) is acceptable.
- [ ] **Rate Limiting**:
    - Use `slowapi` (or similar) to limit requests to `/api/v1/ask` (e.g., 5 per minute per IP).
    - Limit `/api/v1/search` to prevent scraping (e.g., 60 per minute).
- [ ] **Result Limits**:
    - Hard limit `limit` parameter in search API (max 50 or 100).
    - Hard limit context size for RAG (max 5 chunks).

## Acceptance Criteria
- Exceeding rate limits returns 429.
- Exceeding token budget returns 429 or specific error.
- Logs show token usage per request.
