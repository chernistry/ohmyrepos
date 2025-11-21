'use client';
import { RepoCard, RepoProps } from './RepoCard';
import { AnimatePresence, motion } from 'framer-motion';

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
    // Simple client-side filter for mock data
    const filteredRepos = MOCK_REPOS.filter(repo =>
        repo.name.toLowerCase().includes(query.toLowerCase()) ||
        repo.description.toLowerCase().includes(query.toLowerCase())
    );

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-20">
            <AnimatePresence mode='popLayout'>
                {filteredRepos.map((repo, index) => (
                    <RepoCard key={repo.id} repo={repo} index={index} />
                ))}
            </AnimatePresence>

            {filteredRepos.length === 0 && (
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
