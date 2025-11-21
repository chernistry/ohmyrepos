# Ticket 02: Search & Repo Grid

**Goal**: Create the main "Search" view with a high-performance, animated grid of repositories.

## Context
This is the core view. It needs to feel "alive". When results load, they should stagger-fade in. The search bar should feel like a Command Palette.

## Requirements

### 1. Search Input (`components/SearchInput.tsx`)
- [ ] **Visuals**: Large, floating or top-sticky input. Glass background.
- [ ] **Interaction**: Focus expands it slightly or adds a glow ring.
- [ ] **Icon**: Search icon on left, "CMD+K" badge on right.
- [ ] **Debounce**: Implement `useDebounce` hook (300ms) before triggering API.

### 2. Repo Card (`components/RepoCard.tsx`)
- [ ] **Layout**:
  - Top: Owner/Name (Link), Stars (Icon + Count), Language (Color Dot).
  - Middle: Description (truncated to 2 lines).
  - Bottom: Topics (Pills), Last Updated (Relative time).
- [ ] **Style**:
  - `bg-surface` with `border border-border`.
  - Hover: `border-accent/50 shadow-lg scale-[1.01] transition-all`.
- [ ] **Actions**:
  - "Copy URL" button (visible on hover).
  - "Ask" button (triggers chat context).

### 3. Infinite Grid (`components/RepoGrid.tsx`)
- [ ] **Layout**: CSS Grid (`grid-cols-1 md:grid-cols-2 lg:grid-cols-3`).
- [ ] **Animation**: Use `framer-motion` `AnimatePresence` and `motion.div`.
  - `initial={{ opacity: 0, y: 20 }}`
  - `animate={{ opacity: 1, y: 0 }}`
  - `transition={{ delay: index * 0.05 }}`

## Implementation Snippets

**`ui/components/RepoCard.tsx`**:
```tsx
'use client';
import { Star, GitFork, Calendar } from '@phosphor-icons/react';
import { motion } from 'framer-motion';

interface RepoProps {
  name: string;
  description: string;
  stars: number;
  language: string;
  updatedAt: string;
  topics: string[];
}

export function RepoCard({ repo, index }: { repo: RepoProps; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
      className="group relative flex flex-col p-5 rounded-xl bg-surface border border-border hover:border-accent/40 hover:shadow-xl hover:shadow-accent/5 transition-all duration-300 cursor-pointer"
    >
      <div className="flex justify-between items-start mb-3">
        <h3 className="font-semibold text-lg text-primary group-hover:text-accent transition-colors">
          {repo.name}
        </h3>
        <div className="flex items-center gap-1 text-secondary text-sm">
          <Star weight="fill" className="text-yellow-500" />
          <span>{repo.stars}</span>
        </div>
      </div>
      
      <p className="text-secondary text-sm line-clamp-2 mb-4 flex-1">
        {repo.description}
      </p>
      
      <div className="flex flex-wrap gap-2 mt-auto">
        {repo.topics.slice(0, 3).map(topic => (
          <span key={topic} className="px-2 py-0.5 text-xs rounded-full bg-white/5 text-secondary border border-white/5">
            {topic}
          </span>
        ))}
      </div>
      
      <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
        {/* Action buttons here */}
      </div>
    </motion.div>
  );
}
```

**`ui/app/page.tsx` (Search View)**:
```tsx
'use client';
import { useState } from 'react';
import { SearchInput } from '@/components/SearchInput';
import { RepoGrid } from '@/components/RepoGrid';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  
  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight text-white">Discover</h1>
        <p className="text-secondary">Search across your knowledge base.</p>
      </header>
      
      <div className="sticky top-4 z-10">
        <SearchInput value={query} onChange={setQuery} />
      </div>
      
      <RepoGrid query={query} />
    </div>
  );
}
```

## Acceptance Criteria
- Grid renders mock data beautifully.
- Hover effects work.
- Staggered animation works on load.
- Search input is sticky and looks premium.
