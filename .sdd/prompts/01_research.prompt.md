# Best Practices Research Template (Improved)

Instruction for AI: produce a practical, evidence‑backed best practices guide tailored to this project and stack.

---

## Project Context
- Project: ohmyrepos
- Description: # Oh My Repos — Модернизация

## Контекст
Существующий инструмент для семантического поиска по GitHub-репозиториям. Работает локально через CLI/Streamlit, использует BM25 + векторный поиск (Qdrant) + LLM-ранжирование (OpenAI/Ollama) + эмбеддинги (Jina AI).

## Проблемы текущей версии
- Устаревший UI (Streamlit)
- Только локальное использование, нет онлайн-версии
- Неоптимальные затраты на LLM/эмбеддинги
- Архитектура не соответствует современным практикам

## Цели модернизации
- **Онлайн-версия**: Развернуть на бесплатном хостинге (Vercel/Railway/Render)
- **Современный UI**: Заменить Streamlit на веб-интерфейс
- **Оптимизация затрат**: Использовать бесплатные LLM/эмбеддеры где возможно
- **Best practices 2025**: Применить актуальные паттерны для LLM-пайплайнов, RAG, категоризации

## Пользователи
- Разработчики, управляющие большими коллекциями starred/bookmarked репозиториев
- Команды, ищущие референсные проекты по технологиям

## Ключевые ограничения
- Бюджет: минимальный/нулевой (бесплатные тиры)
- Хостинг: бесплатные платформы
- Производительность: поиск должен оставаться быстрым

## Definition of Done
- [ ] Работающая онлайн-версия на бесплатном хостинге
- [ ] Современный веб-UI (не Streamlit)
- [ ] Оптимизированные затраты на LLM/эмбеддинги
- [ ] Документация по архитектуре и деплою
- [ ] Сохранена функциональность: BM25 + векторный поиск + LLM-ранжирование
- Tech stack: Python/FastAPI backend + Next.js frontend + Qdrant
- Domain: LLM-powered semantic search over GitHub repositories
- Year: 2025

## Task
Create a comprehensive best‑practices guide for ohmyrepos that is:
1) Current — relevant to 2025; mark deprecated/outdated items.
2) Specific — tailored to Python/FastAPI backend + Next.js frontend + Qdrant and LLM-powered semantic search over GitHub repositories.
3) Practical — include concrete commands/config/code.
4) Complete — cover architecture, quality, ops, and security.
5) Risk‑aware — define a simple metric profile (PerfGain, SecRisk, DevTime, Maintainability, Cost, DX) with indicative weights for this project, plus 3–5 key risks with High/Medium/Low labels.
6) Verification‑ready — for each major recommendation, note how to validate it (tests, metrics, experiments) so the architect/agent can reuse these checks.

## Output Structure (Markdown)
### 1. TL;DR (≤10 bullets)
- Key decisions and patterns (why, trade‑offs, MVP vs later)
- Observability posture; Security posture; CI/CD; Performance & Cost guardrails
- What changed in 2025; SLOs summary

### 2. Landscape — What’s new in 2025
For Python/FastAPI backend + Next.js frontend + Qdrant:
- Standards/framework updates; deprecations/EOL; pricing changes
- Tooling maturity: testing, observability, security
- Cloud/vendor updates
- Alternative approaches and when to choose them

### 3. Architecture Patterns (2–4 for LLM-powered semantic search over GitHub repositories with Python/FastAPI backend + Next.js frontend + Qdrant)
Pattern A — [NAME] (MVP)
- When to use; Steps; Pros/Cons; Optional later features

Pattern B — [NAME] (Scale‑up)
- When to use; Migration from A

### 4. Priority 1 — [AREA]
Why → relation to goals and mitigated risks
Scope → In/Out
Decisions → with rationale and alternatives
Implementation outline → 3–6 concrete steps
Guardrails & SLOs → metrics and limits/quotas
Failure Modes & Recovery → detection→remediation→rollback

### 5–6. Priority 2/3 — [AREA]
Repeat the structure from 4.

### 7. Testing Strategy (for Python/FastAPI backend + Next.js frontend + Qdrant)
- Unit / Integration / E2E / Performance / Security
- Frameworks, patterns, coverage targets

### 8. Observability & Operations
- Metrics, Logging, Tracing, Alerting, Dashboards

### 9. Security Best Practices
- AuthN/AuthZ, Data protection (PII, encryption), Secrets, Dependency security
- OWASP Top 10 (2025) coverage; Compliance (if any)

### 10. Performance & Cost
- Budgets (concrete numbers), optimization techniques, cost monitoring, resource limits

### 11. CI/CD Pipeline
- Build/Test/Deploy; quality gates; environments

### 12. Code Quality Standards
- Style, linters/formatters, typing, docs, review, refactoring

### 13. Reading List (with dates and gists)
- [Source] (Last updated: YYYY‑MM‑DD) — gist

### 14. Decision Log (ADR style)
- [ADR‑001] [Choice] over [alternatives] because [reason]

### 15. Anti‑Patterns to Avoid
- For Python/FastAPI backend + Next.js frontend + Qdrant/LLM-powered semantic search over GitHub repositories with “what, why bad, what instead”

### 16. Evidence & Citations
- List sources inline near claims; add links; include “Last updated” dates when possible.

### 17. Verification
- Self‑check: how to validate key recommendations (scripts, smoke tests, benchmarks)
- Confidence: [High/Medium/Low] per section

## Requirements
1) No chain‑of‑thought. Provide final answers with short, verifiable reasoning.
2) If browsing is needed, state what to check and why; produce a provisional answer with TODOs.
3) Keep it implementable today; prefer defaults that reduce complexity.
4) Do not fabricate libraries, APIs, or data; if unsure or the evidence is weak, mark the item as TODO/Low confidence and suggest concrete sources to verify.

## Additional Context
{{ADDITIONAL_CONTEXT}}

---
Start the research now and produce the guide for ohmyrepos.