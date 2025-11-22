'use client';
import { RepoCard, RepoProps } from './RepoCard';
import { AnimatePresence, motion } from 'framer-motion';
import { useState, useEffect } from 'react';

// Mock data for now
const MOCK_REPOS: RepoProps[] = [
    {
        id: '1',
        owner: 'vercel',
        name: 'next.js',
        description: 'The React Framework for the Web',
        stars: 115000,
        language: 'TypeScript',
        updatedAt: '2h ago',
        topics: ['react', 'framework', 'web'],
        url: 'https://github.com/vercel/next.js'
    },
    {
        id: '2',
        owner: 'shadcn',
        name: 'ui',
        description: 'Beautifully designed components built with Radix UI and Tailwind CSS.',
        stars: 45000,
        language: 'TypeScript',
        updatedAt: '1d ago',
        topics: ['components', 'radix-ui', 'tailwind'],
        url: 'https://github.com/shadcn-ui/ui'
    },
    {
        id: '3',
        owner: 'qdrant',
        name: 'qdrant',
        description: 'Qdrant - Vector Database for the next generation of AI applications',
        stars: 18000,
        language: 'Rust',
        updatedAt: '5h ago',
        topics: ['vector-database', 'rust', 'search'],
        url: 'https://github.com/qdrant/qdrant'
    },
    {
        id: '4',
        owner: 'python',
        name: 'cpython',
        description: 'The Python programming language',
        stars: 58000,
        language: 'Python',
        updatedAt: '10m ago',
        topics: ['python', 'language', 'c'],
        url: 'https://github.com/python/cpython'
    },
];

interface RepoGridProps {
    query: string;
}

export function RepoGrid({ query }: RepoGridProps) {
    const [repos, setRepos] = useState<RepoProps[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchRepos = async () => {
            setLoading(true);
            try {
                let data;
                if (!query) {
                    // Fetch random repos
                    const res = await fetch('http://127.0.0.1:8000/api/v1/random?limit=12');
                    if (!res.ok) throw new Error('Failed to fetch random repos');
                    data = await res.json();
                } else {
                    // Search repos
                    const res = await fetch('http://127.0.0.1:8000/api/v1/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, limit: 12 })
                    });
                    if (!res.ok) throw new Error('Failed to search repos');
                    const response = await res.json();
                    data = response.results;
                }

                // Map API response to RepoProps
                const mappedRepos = data.map((repo: any, index: number) => ({
                    id: repo.url || `repo-${index}`,
                    owner: repo.full_name.split('/')[0] || 'unknown',
                    name: repo.repo_name,
                    description: repo.summary || repo.description || '',
                    stars: repo.stars,
                    language: repo.language || 'Unknown',
                    updatedAt: 'Recently', // API doesn't return this yet
                    topics: repo.tags || [],
                    url: repo.url
                }));

                setRepos(mappedRepos);
            } catch (error) {
                console.error('Error fetching repositories:', error);
                setRepos([]);
            } finally {
                setLoading(false);
            }
        };

        const timeoutId = setTimeout(fetchRepos, query ? 500 : 0);
        return () => clearTimeout(timeoutId);
    }, [query]);

    if (loading) {
        return (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-20">
                {[...Array(6)].map((_, i) => (
                    <div key={i} className="h-64 bg-white/5 rounded-xl animate-pulse" />
                ))}
            </div>
        );
    }

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-20">
            <AnimatePresence mode='popLayout'>
                {repos.map((repo, index) => (
                    <RepoCard key={repo.id} repo={repo} index={index} />
                ))}
            </AnimatePresence>

            {repos.length === 0 && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="col-span-full text-center py-20 text-secondary"
                >
                    No repositories found matching "{query}"
                </motion.div>
            )}
        </div>
    );
}
