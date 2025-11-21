'use client';
import { useState, useRef, useEffect } from 'react';
import { PaperPlaneRight, Stop, Trash, Robot } from '@phosphor-icons/react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChat, Message } from '@/hooks/useChat';
import { clsx } from 'clsx';
import ReactMarkdown from 'react-markdown';

interface ChatInterfaceProps {
    isOpen: boolean;
    onClose: () => void;
}

export function ChatInterface({ isOpen, onClose }: ChatInterfaceProps) {
    const { messages, sendMessage, isLoading, stop, clear } = useChat();
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;
        sendMessage(input);
        setInput('');
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0, x: '100%' }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: '100%' }}
                    transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                    className="fixed right-0 top-0 bottom-0 w-[400px] bg-surface border-l border-border shadow-2xl z-40 flex flex-col"
                >
                    {/* Header */}
                    <div className="p-4 border-b border-border flex justify-between items-center bg-background/50 backdrop-blur">
                        <div className="flex items-center gap-2">
                            <div className="p-1.5 bg-accent/10 rounded-lg text-accent">
                                <Robot size={20} weight="duotone" />
                            </div>
                            <h2 className="font-semibold">Ask AI</h2>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={clear}
                                className="p-1.5 text-secondary hover:text-red-400 hover:bg-red-400/10 rounded-md transition-colors"
                                title="Clear chat"
                            >
                                <Trash size={18} />
                            </button>
                            <button
                                onClick={onClose}
                                className="p-1.5 text-secondary hover:text-primary hover:bg-white/10 rounded-md transition-colors"
                            >
                                Close
                            </button>
                        </div>
                    </div>

                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-6">
                        {messages.length === 0 && (
                            <div className="text-center text-secondary mt-20 space-y-2">
                                <Robot size={48} className="mx-auto opacity-20" />
                                <p>Ask anything about your repositories.</p>
                            </div>
                        )}

                        {messages.map((msg, idx) => (
                            <div key={idx} className={clsx("flex gap-3", msg.role === 'user' ? "flex-row-reverse" : "")}>
                                <div className={clsx(
                                    "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                                    msg.role === 'user' ? "bg-primary text-background" : "bg-accent/10 text-accent"
                                )}>
                                    {msg.role === 'user' ? 'U' : <Robot size={16} />}
                                </div>
                                <div className={clsx(
                                    "max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
                                    msg.role === 'user'
                                        ? "bg-primary text-background rounded-tr-none"
                                        : "bg-white/5 border border-white/5 rounded-tl-none"
                                )}>
                                    <ReactMarkdown className="prose prose-invert prose-sm max-w-none">
                                        {msg.content}
                                    </ReactMarkdown>
                                </div>
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <div className="p-4 border-t border-border bg-background/50 backdrop-blur">
                        <form onSubmit={handleSubmit} className="relative">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask a question..."
                                className="w-full bg-surface border border-border rounded-xl pl-4 pr-12 py-3 focus:ring-1 focus:ring-accent outline-none transition-all"
                                disabled={isLoading}
                            />
                            <div className="absolute right-2 top-1/2 -translate-y-1/2">
                                {isLoading ? (
                                    <button
                                        type="button"
                                        onClick={stop}
                                        className="p-1.5 bg-red-500/10 text-red-500 rounded-lg hover:bg-red-500/20 transition-colors"
                                    >
                                        <Stop size={18} weight="fill" />
                                    </button>
                                ) : (
                                    <button
                                        type="submit"
                                        disabled={!input.trim()}
                                        className="p-1.5 bg-accent text-white rounded-lg hover:bg-accent/90 disabled:opacity-50 disabled:hover:bg-accent transition-colors"
                                    >
                                        <PaperPlaneRight size={18} weight="fill" />
                                    </button>
                                )}
                            </div>
                        </form>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
