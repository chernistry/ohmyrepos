Task: Score a GitHub repository based on relevance to the user's interest profile.

User Interest Profile:
- Cluster: {{profile_name}}
- Keywords: {{profile_keywords}}

Repository:
- Name: {{repo_full_name}}
- Description: {{repo_description}}
- Language: {{repo_language}}
- Stars: {{repo_stars}}

README Content:
{{readme_content}}

Instructions:
- Evaluate the repository on:
  1. Relevance to user interests (keywords, domain)
  2. Code quality indicators (documentation, structure)
  3. Project maturity (stars, activity, completeness)
- Provide a score from 0.0 to 10.0
- Include brief reasoning for the score
- Optionally provide a summary of the repository

Output JSON format:
```json
{
  "score": 8.5,
  "reasoning": "Brief explanation of the score",
  "summary": "Optional brief summary of what the repo does"
}
```
