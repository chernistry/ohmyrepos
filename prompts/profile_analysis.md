Task: Analyze the user's starred repositories and identify interest clusters.

Repositories:
{{repos_text}}

Instructions:
- Identify 3-5 distinct interest clusters based on common themes, technologies, or domains
- For each cluster, extract:
  - A descriptive name
  - Key technical keywords
  - Primary programming languages
  - A relevance score (0.0 - 1.0)

Output JSON format:
```json
{
  "clusters": [
    {
      "name": "Cluster Name",
      "keywords": ["keyword1", "keyword2"],
      "languages": ["Python", "JavaScript"],
      "score": 0.95
    }
  ]
}
```
