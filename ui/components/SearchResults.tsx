import { RepoResult } from '@/lib/api';
import RepoCard from './RepoCard';

interface SearchResultsProps {
  results: RepoResult[];
  query: string;
}

export default function SearchResults({ results, query }: SearchResultsProps) {
  if (results.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No results found for &quot;{query}&quot;
      </div>
    );
  }

  return (
    <div className="w-full max-w-4xl">
      <div className="mb-4 text-gray-600">
        Found {results.length} repositories
      </div>
      <div className="space-y-4">
        {results.map((repo) => (
          <RepoCard key={repo.full_name} repo={repo} />
        ))}
      </div>
    </div>
  );
}
