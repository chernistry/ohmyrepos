'use client';

import { useState } from 'react';
import SearchBar from '@/components/SearchBar';
import SearchResults from '@/components/SearchResults';
import { searchRepos, RepoResult } from '@/lib/api';

export default function Home() {
  const [results, setResults] = useState<RepoResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState('');

  const handleSearch = async (query: string) => {
    setIsLoading(true);
    setError(null);
    setLastQuery(query);

    try {
      const response = await searchRepos(query);
      setResults(response.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="container mx-auto px-4 py-8">
        <div className="flex flex-col items-center gap-8">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">
              Oh My Repos
            </h1>
            <p className="text-gray-600">
              Semantic search for GitHub repositories
            </p>
          </div>

          <SearchBar onSearch={handleSearch} isLoading={isLoading} />

          {error && (
            <div className="w-full max-w-3xl p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
              {error}
            </div>
          )}

          {!isLoading && lastQuery && (
            <SearchResults results={results} query={lastQuery} />
          )}

          {isLoading && (
            <div className="text-gray-500">Searching...</div>
          )}
        </div>
      </main>
    </div>
  );
}
