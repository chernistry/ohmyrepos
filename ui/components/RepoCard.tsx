'use client';
import { Star, GitFork, Calendar, ArrowSquareOut, ChatCircleDots } from '@phosphor-icons/react';
import { motion } from 'framer-motion';

export interface RepoProps {
  id: string;
  name: string;
  owner: string;
  description: string;
  stars: number;
  language: string;
  updatedAt: string;
  topics: string[];
  url: string;
}

const LANGUAGE_COLORS: Record<string, string> = {
  Python: '#3572A5',
  TypeScript: '#2b7489',
  JavaScript: '#f1e05a',
  Rust: '#dea584',
  Go: '#00ADD8',
  HTML: '#e34c26',
  CSS: '#563d7c',
};

export function RepoCard({ repo, index }: { repo: RepoProps; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
      className="group relative flex flex-col p-5 rounded-xl bg-surface border border-border hover:border-accent/40 hover:shadow-xl hover:shadow-accent/5 transition-all duration-300 cursor-pointer"
    >
      <div className="flex justify-between items-start mb-3">
        <div className="flex flex-col">
          <span className="text-xs text-secondary mb-0.5">{repo.owner}</span>
          <h3 className="font-semibold text-lg text-primary group-hover:text-accent transition-colors">
            {repo.name}
          </h3>
        </div>
        <div className="flex items-center gap-1 text-secondary text-sm bg-white/5 px-2 py-1 rounded-full">
          <Star weight="fill" className="text-yellow-500" />
          <span>{repo.stars}</span>
        </div>
      </div>

      <p className="text-secondary text-sm line-clamp-2 mb-4 flex-1 leading-relaxed">
        {repo.description}
      </p>

      <div className="flex items-center gap-4 mb-4 text-xs text-secondary">
        <div className="flex items-center gap-1.5">
          <span
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: LANGUAGE_COLORS[repo.language] || '#ccc' }}
          />
          <span>{repo.language}</span>
        </div>
        <div className="flex items-center gap-1">
          <Calendar size={14} />
          <span>{repo.updatedAt}</span>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 mt-auto">
        {repo.topics.slice(0, 3).map(topic => (
          <span key={topic} className="px-2 py-0.5 text-xs rounded-full bg-white/5 text-secondary border border-white/5 group-hover:border-white/10 transition-colors">
            {topic}
          </span>
        ))}
        {repo.topics.length > 3 && (
          <span className="px-2 py-0.5 text-xs rounded-full text-secondary/50">
            +{repo.topics.length - 3}
          </span>
        )}
      </div>

      {/* Hover Actions */}
      <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
        <button
          className="p-2 rounded-lg bg-background/80 backdrop-blur text-secondary hover:text-accent hover:bg-accent/10 border border-border hover:border-accent/30 transition-colors"
          title="Ask about this repo"
        >
          <ChatCircleDots size={18} />
        </button>
        <a
          href={repo.url}
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-lg bg-background/80 backdrop-blur text-secondary hover:text-primary hover:bg-white/10 border border-border hover:border-white/20 transition-colors"
          title="Open in GitHub"
        >
          <ArrowSquareOut size={18} />
        </a>
      </div>
    </motion.div>
  );
}
