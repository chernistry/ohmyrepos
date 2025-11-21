import { RepoResult } from '@/lib/api';

interface RepoCardProps {
  repo: RepoResult;
}

export default function RepoCard({ repo }: RepoCardProps) {
  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-lg transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <a
          href={repo.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xl font-semibold text-blue-600 hover:underline"
        >
          {repo.full_name}
        </a>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          {repo.language && (
            <span className="px-2 py-1 bg-gray-100 rounded">{repo.language}</span>
          )}
          <span>⭐ {repo.stars}</span>
        </div>
      </div>
      
      {repo.description && (
        <p className="text-gray-700 mb-2">{repo.description}</p>
      )}
      
      {repo.summary && (
        <p className="text-sm text-gray-600 mb-3 italic">{repo.summary}</p>
      )}
      
      {repo.tags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {repo.tags.map((tag) => (
            <span
              key={tag}
              className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded"
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
