Task: Generate optimized GitHub search queries from a natural language intent.

Intent: "{{intent}}"
Strategy: {{strategy}}

Instructions:
- For "specific" strategy: Generate 3-5 precise, targeted queries using exact technical terms
- For "broad" strategy: Generate 3-5 wider discovery queries with related concepts

Output JSON format:
```json
{
  "queries": ["query1", "query2", "query3"]
}
```
