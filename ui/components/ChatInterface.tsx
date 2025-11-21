'use client';
import { useState, useRef, useEffect } from 'react';
import { PaperPlaneRight, Trash, Robot, X, User, Lightbulb, Code, Sparkle } from '@phosphor-icons/react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChat } from '@/hooks/useChat';
import { clsx } from 'clsx';
import ReactMarkdown from 'react-markdown';

interface ChatInterfaceProps {
    isOpen: boolean;
    onClose: () => void;
}

const SUGGESTIONS = [
    { text: "Summarize this repo", icon: Sparkle },
    { text: "Explain the code structure", icon: Code },
    { text: "How do I contribute?", icon: User },
    { text: "Find similar projects", icon: Lightbulb }
];

export function ChatInterface({ isOpen, onClose }: ChatInterfaceProps) {
    const { messages, sendMessage, isLoading, stop, clear } = useChat();
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
        }
    }, [input]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;
        sendMessage(input);
        setInput('');
        if (textareaRef.current) textareaRef.current.style.height = 'auto';
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
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
                        className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
                    />
                    <motion.div
                        initial={{ opacity: 0, x: '100%' }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: '100%' }}
                        transition={{ type: 'spring', damping: 30, stiffness: 300 }}
                        className="fixed right-0 top-0 bottom-0 w-full max-w-[500px] bg-background/95 backdrop-blur-xl border-l border-white/10 shadow-2xl z-50 flex flex-col"
                    >
                        {/* Header */}
                        <div className="p-4 border-b border-white/5 flex justify-between items-center bg-white/5 backdrop-blur-md">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-accent/10 rounded-xl text-accent border border-accent/20">
                                    <Robot size={24} weight="duotone" />
                                </div>
                                <div>
                                    <h2 className="font-semibold text-lg tracking-tight">Ask AI</h2>
                                    <p className="text-xs text-secondary">Powered by LLM</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={clear}
                                    className="p-2 text-secondary hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-colors"
                                    title="Clear chat"
                                >
                                    <Trash size={20} />
                                </button>
                                <button
                                    onClick={onClose}
                                    className="p-2 text-secondary hover:text-primary hover:bg-white/10 rounded-lg transition-colors"
                                >
                                    <X size={20} />
                                </button>
                            </div>
                        </div>

                        {/* Messages */}
                        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide">
                            {messages.length === 0 && (
                                <div className="h-full flex flex-col justify-center items-center">
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="text-center space-y-6 max-w-sm"
                                    >
                                        <div className="w-20 h-20 mx-auto bg-gradient-to-br from-accent/20 to-accent/5 rounded-3xl flex items-center justify-center text-accent border border-accent/10 shadow-premium">
                                            <Sparkle size={40} weight="fill" className="animate-pulse-slow" />
                                        </div>
                                        <h3 className="text-2xl font-bold text-primary">How can I help?</h3>

                                        <div className="grid grid-cols-1 gap-3 w-full">
                                            {SUGGESTIONS.map((suggestion, idx) => (
                                                <button
                                                    key={idx}
                                                    onClick={() => {
                                                        sendMessage(suggestion.text);
                                                    }}
                                                    className="flex items-center gap-3 p-4 text-left bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/10 rounded-xl transition-all duration-200 group"
                                                >
                                                    <div className="p-2 bg-white/5 rounded-lg text-secondary group-hover:text-primary transition-colors">
                                                        <suggestion.icon size={18} weight="duotone" />
                                                    </div>
                                                    <span className="text-sm font-medium text-secondary group-hover:text-primary transition-colors">
                                                        {suggestion.text}
                                                    </span>
                                                </button>
                                            ))}
                                        </div>
                                    </motion.div>
                                </div>
                            )}

                            {messages.map((msg, idx) => (
                                <motion.div
                                    key={idx}
                                    initial={{ opacity: 0, y: 10, scale: 0.98 }}
                                    animate={{ opacity: 1, y: 0, scale: 1 }}
                                    className={clsx("flex gap-4", msg.role === 'user' ? "flex-row-reverse" : "")}
                                >
                                    <div className={clsx(
                                        "w-10 h-10 rounded-xl flex items-center justify-center shrink-0 shadow-sm",
                                        msg.role === 'user' ? "bg-primary text-background" : "bg-accent/10 text-accent border border-accent/20"
                                    )}>
                                        {msg.role === 'user' ? <User size={20} weight="bold" /> : <Robot size={20} weight="duotone" />}
                                    </div>
                                    <div className={clsx(
                                        "max-w-[85%] rounded-2xl px-6 py-4 text-sm leading-relaxed shadow-sm",
                                        msg.role === 'user'
                                            ? "bg-white/10 text-primary rounded-tr-sm border border-white/10"
                                            : "bg-black/40 text-secondary/90 rounded-tl-sm border border-white/5 backdrop-blur-sm"
                                    )}>
                                        <div className="prose prose-invert prose-sm max-w-none">
                                            <ReactMarkdown>{msg.content}</ReactMarkdown>
                                        </div>
                                    </div>
                                </motion.div>
                            ))}

                            {isLoading && (
                                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-4">
                                    <div className="w-10 h-10 rounded-xl bg-accent/10 text-accent border border-accent/20 flex items-center justify-center shrink-0">
                                        <Robot size={20} weight="duotone" />
                                    </div>
                                    <div className="bg-black/40 border border-white/5 rounded-2xl rounded-tl-sm px-6 py-4 flex items-center gap-1.5">
                                        <div className="w-2 h-2 bg-secondary/50 rounded-full animate-bounce" style={{ animationDuration: '0.6s' }} />
                                        <div className="w-2 h-2 bg-secondary/50 rounded-full animate-bounce" style={{ animationDuration: '0.6s', animationDelay: '0.2s' }} />
                                        <div className="w-2 h-2 bg-secondary/50 rounded-full animate-bounce" style={{ animationDuration: '0.6s', animationDelay: '0.4s' }} />
                                    </div>
                                </motion.div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input */}
                        <div className="p-6 border-t border-white/5 bg-background/80 backdrop-blur-xl">
                            <form onSubmit={handleSubmit} className="relative group">
                                <div className="relative flex items-end gap-2 p-2 rounded-2xl bg-white/5 border border-white/10 shadow-inner focus-within:bg-white/10 focus-within:border-accent/30 focus-within:ring-1 focus-within:ring-accent/30 transition-all duration-300">
                                    <textarea
                                        ref={textareaRef}
                                        value={input}
                                        onChange={(e) => setInput(e.target.value)}
                                        onKeyDown={handleKeyDown}
                                        placeholder="Ask anything..."
                                        className="flex-1 resize-none bg-transparent border-none px-4 py-3 text-primary placeholder:text-secondary/50 focus:ring-0 outline-none min-h-[52px] max-h-[200px] scrollbar-hide"
                                        rows={1}
                                        disabled={isLoading}
                                    />
                                    <button
                                        type="submit"
                                        disabled={!input.trim() || isLoading}
                                        className="mb-1.5 mr-1.5 p-2.5 rounded-xl bg-accent text-white hover:bg-accent/90 disabled:opacity-50 disabled:hover:bg-accent transition-all duration-200 shadow-lg shadow-accent/20"
                                    >
                                        <PaperPlaneRight size={20} weight="fill" />
                                    </button>
                                </div>
                                <p className="text-center text-xs text-secondary/40 mt-3">
                                    AI can make mistakes. Please verify important information.
                                </p>
                            </form>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
