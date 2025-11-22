Task: Generate a NEW, DISTINCT search intent to discover more GitHub repositories related to the user's original goal.

Original Goal: "{{original_intent}}"

We have already searched for:
{{previous_intents}}

We have already found these repositories (DO NOT search for them again):
{{found_repos}}

Goal: Find *different* or *niche* or *related* repositories that we might have missed.

Think about:
- Alternative frameworks or libraries
- Specific use cases (e.g., "for finance", "for healthcare")
- Underlying technologies (e.g., "vector database", "knowledge graph")
- Competitors or alternatives to found repos

Output JSON format:
```json
{
  "new_intent": "your new search query string"
}
```
