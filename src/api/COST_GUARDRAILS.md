# Cost Guardrails

Protection mechanisms to prevent cost overruns and abuse.

## Features

### 1. Token Budget Tracking

Tracks daily token usage against a configurable limit.

**Configuration:**
```env
MAX_DAILY_TOKENS=1000000  # Optional, defaults to unlimited
```

**Behavior:**
- Estimates tokens for each request (~4 chars = 1 token)
- Tracks cumulative daily usage
- Resets automatically after 24 hours
- Returns 429 when budget exceeded

**Response when exceeded:**
```json
{
  "code": "budget_exceeded",
  "message": "Daily token budget exceeded. Remaining: 0"
}
```

### 2. Rate Limiting

Per-IP rate limiting for API endpoints.

**Limits:**
- `/api/v1/search`: 60 requests per minute
- Future `/api/v1/ask`: 5 requests per minute (when implemented)

**Behavior:**
- Tracks requests per IP per endpoint
- Sliding window implementation
- Automatic cleanup of old records

**Response when exceeded:**
```json
{
  "code": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Maximum 60 requests per 60 seconds."
}
```

### 3. Result Limits

Hard limits on response sizes.

**Limits:**
- Search `limit` parameter: max 100 results
- Validated at request level via Pydantic

## Implementation

### Token Estimation

```python
from src.api.utils import estimate_tokens, log_token_usage

tokens = estimate_tokens("your text here")
log_token_usage("operation_name", input_tokens=tokens)
```

### Budget Tracking

```python
from src.api.budget import get_budget_tracker

tracker = get_budget_tracker()

# Check if request is within budget
if not tracker.check_budget(estimated_tokens):
    raise HTTPException(status_code=429, detail="Budget exceeded")

# Add usage after successful request
tracker.add_usage(actual_tokens)
```

### Rate Limiting

```python
from fastapi import Depends
from src.api.rate_limit import rate_limit_dependency

@router.post(
    "/endpoint",
    dependencies=[Depends(lambda req: rate_limit_dependency(req, limit=60, window=60))]
)
async def endpoint():
    pass
```

## Monitoring

All guardrail events are logged with structured logging:

```json
{
  "event": "budget_usage",
  "daily_usage": 50000,
  "max_daily_tokens": 1000000,
  "percentage_used": 5.0
}
```

```json
{
  "event": "rate_limit_exceeded",
  "client_ip": "192.168.1.1",
  "endpoint": "/api/v1/search",
  "count": 61,
  "limit": 60
}
```

## Testing

```bash
pytest tests/test_cost_guardrails.py -v
```

## Production Considerations

1. **Token Counting**: Current implementation uses a simple heuristic. For production, consider:
   - Using `tiktoken` for accurate OpenAI token counting
   - Model-specific token counters

2. **Budget Storage**: Current implementation uses in-memory storage. For production:
   - Use Redis or database for persistent tracking
   - Implement distributed counters for multi-instance deployments

3. **Rate Limiting**: Current implementation is in-memory. For production:
   - Use Redis with sliding window algorithm
   - Consider using a dedicated service like Kong or Nginx rate limiting

4. **Monitoring**: Set up alerts for:
   - Budget usage > 80%
   - Rate limit violations > threshold
   - Unusual traffic patterns
