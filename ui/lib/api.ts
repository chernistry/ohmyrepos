const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export interface RepoResult {
  repo_name: string;
  full_name: string;
  description: string | null;
  summary: string | null;
  tags: string[];
  language: string | null;
  stars: number;
  url: string;
  score: number;
}

export interface SearchResponse {
  query: string;
  results: RepoResult[];
  total: number;
}

export async function searchRepos(
  query: string,
  limit: number = 25,
  filterTags?: string[]
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit, filter_tags: filterTags }),
  });

  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`);
  }

  return response.json();
}
