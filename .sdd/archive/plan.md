# Oh My Repos — Development Plan (v4)

## Mission
Создать локальный инструмент для каталогизации и интеллектуального поиска по starred-репозиториям GitHub, используя модульную архитектуру для LLM и эмбеддингов, гибридный поиск и кластеризацию.

## Guiding Principles
*   **Модульная архитектура**: Адаптируем `core/llm` и `core/embeddings` из `@langgraph`.
*   **Python 3.12+**, `Ruff`/`Black`.
*   **Технологический стек**:
    *   **LLM-интеграция**: Модуль `llm` с фабрикой и провайдерами (OpenAI, Ollama).
    *   **Эмбеддинги**: Модуль `core.embeddings` с фабрикой (Jina AI по умолчанию).
    *   **Векторная база**: Qdrant Cloud.
    *   **Поиск**: Гибридный (Dense/Qdrant + Sparse/BM25).
    *   **Ранжирование**: Jina AI (`jina-reranker-v1-base-en`).
    *   **UI**: Streamlit.
*   **Фокус на локальной разработке**.

## High-Level Pipeline
1.  **Collect**: Сбор starred-репозиториев (`collector.py`). ✅
2.  **Summarize**: Генерация структурированного саммари (описание + теги) с помощью `llm.generator` (`summarizer.py`). ✅
3.  **Embed & Store**: Генерация эмбеддингов из саммари с помощью `embeddings.factory`. Загрузка в Qdrant (`storage.py`). ✅
4.  **Cluster**: Кластеризация векторов, именование кластеров с помощью LLM (`clusterer.py`).
5.  **Search & Rerank**: Гибридный поиск и ранжирование (`retriever.py`, `reranker.py`).
6.  **Serve**: Отображение в Streamlit UI (`app.py`).

## Proposed Folder Layout
```
ohmyrepos/
├── .dev/
│   ├── plan.md
│   └── start_prompt.md
├── prompts/
│   └── summarize_repo.md
├── src/
│   └────── __init__.py
│       ├── config.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── collector.py
│       │   ├── summarizer.py
│       │   ├── storage.py
│       │   ├── clusterer.py
│       │   ├── retriever.py
│       │   ├── reranker.py
│       │   └── embeddings/       # Адаптированный модуль из @langgraph
│       │       ├── __init__.py
│       │       ├── base.py
│       │       ├── factory.py
│       │       └── providers/
│       │           ├── __init__.py
│       │           └── jina.py
│       ├── llm/                # Адаптированный модуль из @langgraph
│       │   ├── __init__.py
│       │   ├── chat_adapter.py
│       │   ├── generator.py
│       │   ├── prompt_builder.py
│       │   ├── reply_extractor.py
│       │   └── providers/
│       │       ├── __init__.py
│       │       ├── base.py
│       │       ├── openai.py
│       │       └── ollama.py
│       ├── app.py
│       └── cli.py
├── scripts/
│   └── test_ollama.py
├── tests/
├── ohmyrepos.py
├── requirements.txt
└── README.md
```

## Sprint Roadmap (5 итераций)
| # | Цель | Основные задачи | Результат |
|---|------|-----------------|-----------|
| 1 | **Архитектура и LLM-ядро** ✅ | • Создать **полную** структуру проекта согл. `Folder Layout`.<br>• Адаптировать и интегрировать модуль `llm`.<br>• Реализовать `config.py` и `prompts/summarize_repo.md`.<br>• Реализовать `core/summarizer.py`.<br>• CLI-команда `summarize` для теста. | Рабочий модуль LLM и сервис для обогащения данных. |
| 2 | **Сбор и векторное хранилище** ✅ | • Реализовать `core/collector.py`.<br>• Адаптировать модуль `core/embeddings` из `@langgraph`.<br>• Интегрировать `QdrantStore` и `embeddings.factory` в `core/storage.py`.<br>• CLI-команда `embed` для полного цикла.<br>• Добавить поддержку Ollama для локальных моделей.<br>• Создать скрипт-обертку для запуска CLI.<br>• Добавить команду `generate-summary` для дебага. | Векторная база, наполненная данными. |
| 3 | **Поиск и ранжирование** | • Настроить `HybridRetriever` в `core/retriever.py`.<br>• Реализовать `JinaReranker` в `core/reranker.py`.<br>• CLI-команда `search "query"`.<br>• Добавить поддержку фильтрации по тегам. | Рабочий механизм поиска и ранжирования. |
| 4 | **Кластеризация и UI** | • Реализовать `core/clusterer.py`.<br>• CLI-команда `cluster`.<br>• Создать базовый UI в `app.py`. | Интерактивное приложение с поиском. |
| 5 | **Интеграция и полировка** | • Связать все шаги в `cli.py` (команда `run-all`).<br>• Улучшить UI для отображения кластеров.<br>• Добавить тесты и документацию. | Готовый к локальному использованию продукт. |
