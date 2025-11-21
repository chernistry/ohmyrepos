'use client';
import { useState } from 'react';
import { X, Plus, GitBranch } from '@phosphor-icons/react';
import { motion, AnimatePresence } from 'framer-motion';
import { useIngestion } from '@/hooks/useIngestion';

interface AddRepoModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export function AddRepoModal({ isOpen, onClose }: AddRepoModalProps) {
    const [url, setUrl] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const { addRepo } = useIngestion();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!url) return;

        setIsLoading(true);
        const success = await addRepo(url);
        setIsLoading(false);

        if (success) {
            setUrl('');
            onClose();
        }
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
                    />
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md bg-surface border border-border rounded-xl shadow-2xl z-50 p-6"
                    >
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-xl font-semibold flex items-center gap-2">
                                <GitBranch className="text-accent" />
                                Add Repository
                            </h2>
                            <button onClick={onClose} className="text-secondary hover:text-primary">
                                <X size={20} />
                            </button>
                        </div>

                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-secondary mb-1">
                                    Repository URL or owner/name
                                </label>
                                <input
                                    type="text"
                                    value={url}
                                    onChange={(e) => setUrl(e.target.value)}
                                    placeholder="https://github.com/owner/repo"
                                    className="w-full bg-background border border-border rounded-lg px-3 py-2 focus:ring-1 focus:ring-accent outline-none transition-all"
                                    autoFocus
                                />
                            </div>

                            <div className="flex justify-end gap-3 pt-2">
                                <button
                                    type="button"
                                    onClick={onClose}
                                    className="px-4 py-2 text-sm font-medium text-secondary hover:text-primary"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    disabled={isLoading || !url}
                                    className="px-4 py-2 text-sm font-medium bg-primary text-background rounded-lg hover:bg-white/90 disabled:opacity-50 flex items-center gap-2"
                                >
                                    {isLoading ? 'Adding...' : 'Add Repository'}
                                </button>
                            </div>
                        </form>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
