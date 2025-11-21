# Ticket 05: Observability & Healthchecks

**Goal**: Implement logging, monitoring, and health checks to ensure the system is observable and reliable in production.

## Context
We need to know when the system is down or slow, and we need to debug issues without SSH-ing into containers.

## Requirements
- [ ] **Structured Logging**:
    - Configure `structlog` (or Python's `logging` with a JSON formatter) for the FastAPI app.
    - Ensure logs include: `timestamp`, `level`, `request_id`, `path`, `status_code`, `duration_ms`.
    - Ensure no sensitive data (API keys) is logged.
- [ ] **Health Endpoints**:
    - Enhance `/healthz`: fast, no dependencies (already in Ticket 02, verify it).
    - Enhance `/readyz`: checks Qdrant connectivity and OpenRouter availability (optional).
- [ ] **Metrics (MVP)**:
    - Log-based metrics are sufficient for MVP (e.g., "Search request completed in 150ms").
    - Optional: Add `prometheus-fastapi-instrumentator` if we have a way to scrape it (likely not in MVP free tier, so stick to logs).
- [ ] **Error Handling**:
    - Global exception handler in FastAPI to catch unhandled exceptions and log them with stack traces (structured).
    - Return user-friendly JSON errors (e.g., `{"error": {"code": "internal_error", "message": "..."}}`).

## Acceptance Criteria
- Logs are output in JSON format.
- Request ID is propagated through the request lifecycle.
- `/readyz` correctly reports status of dependencies.
- 500 errors are logged with stack traces but return safe JSON to client.
