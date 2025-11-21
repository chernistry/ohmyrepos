'use client';
import { useState } from 'react';
import { SearchInput } from '@/components/SearchInput';
import { RepoGrid } from '@/components/RepoGrid';
import { useDebounce } from '@/hooks/useDebounce';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-12">
      <header className="space-y-4 text-center mt-10">
        <h1 className="text-4xl font-bold tracking-tight text-primary bg-gradient-to-b from-white to-white/60 bg-clip-text text-transparent">
          Discover
        </h1>
        <p className="text-secondary max-w-lg mx-auto text-lg">
          Search across your knowledge base with semantic understanding.
        </p>
      </header>

      <div className="sticky top-4 z-10">
        <SearchInput value={query} onChange={setQuery} />
      </div>

      <RepoGrid query={debouncedQuery} />
    </div>
  );
}
