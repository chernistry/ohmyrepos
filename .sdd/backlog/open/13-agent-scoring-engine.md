# Ticket 13: Agent Scoring Engine

**Goal**: Implement Stage 4 & 5: LLM-based quality scoring and verification.

## Requirements
-   **Fetch Content**: Retrieve README for candidate repos.
-   **LLM Scoring**:
    -   Design a prompt to evaluate repo relevance to the user profile.
    -   Score on: Freshness, Activity, Relevance, Quality.
    -   Output a score (0-10) and reasoning.
-   **Verification**: Filter out low scores (< 7/10).

## Implementation Details
-   **Module**: `src/agent/scoring.py`
-   **Function**: `score_candidates(candidates: List[Repo], profile: Profile) -> List[ScoredRepo]`
-   **Prompt**: Use best practices (CoT, few-shot if needed).

## Verification
-   Test with a known "good" repo and "bad" repo against a specific profile.
-   Verify LLM output parsing (JSON or structured).
