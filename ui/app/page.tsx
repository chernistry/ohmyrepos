'use client';
import { useState } from 'react';
import { SearchInput } from '@/components/SearchInput';
import { RepoGrid } from '@/components/RepoGrid';
import { ChatInterface } from '@/components/ChatInterface';
import { useDebounce } from '@/hooks/useDebounce';
import { Robot } from '@phosphor-icons/react';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const debouncedQuery = useDebounce(query, 300);

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
          <SearchInput value={query} onChange={setQuery} />
        </div>
        <button
          onClick={() => setIsChatOpen(!isChatOpen)}
          className="h-[58px] px-4 bg-surface border border-border rounded-xl hover:border-accent/50 hover:text-accent transition-colors flex items-center gap-2 group"
        >
          <Robot size={24} weight="duotone" className="group-hover:animate-bounce" />
          <span className="font-medium hidden sm:inline">Ask AI</span>
        </button>
      </div>

      <RepoGrid query={debouncedQuery} />

      <ChatInterface isOpen={isChatOpen} onClose={() => setIsChatOpen(false)} />
    </div>
  );
}
