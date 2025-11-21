'use client';
import { useState } from 'react';
import { SearchInput } from '@/components/SearchInput';
import { RepoGrid } from '@/components/RepoGrid';
import { ChatInterface } from '@/components/ChatInterface';
import { useDebounce } from '@/hooks/useDebounce';
import { Robot } from '@phosphor-icons/react';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [chatQuery, setChatQuery] = useState<string | undefined>(undefined);
  const debouncedQuery = useDebounce(query, 300);

  const handleSearchSubmit = () => {
    if (query.trim()) {
      setChatQuery(query);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-12 relative">
      <header className="space-y-4 text-center mt-10">
        <h1 className="text-4xl font-bold tracking-tight text-primary bg-gradient-to-b from-white to-white/60 bg-clip-text text-transparent">
          Discover
        </h1>
        <p className="text-secondary max-w-lg mx-auto text-lg">
          Search across your knowledge base with semantic understanding.
        </p>
      </header>

      <div className="sticky top-4 z-10 flex justify-center gap-4">
        <div className="flex-1 max-w-2xl">
          <SearchInput
            value={query}
            onChange={(v) => {
              setQuery(v);
              // If user clears input, maybe clear chat query? 
              // Or keep it? Let's keep it for now.
            }}
            onSubmit={handleSearchSubmit}
          />
        </div>
      </div>

      {chatQuery && (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          <ChatInterface initialQuery={chatQuery} />
        </div>
      )}

      <RepoGrid query={debouncedQuery} />
    </div>
  );
}
