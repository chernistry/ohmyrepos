'use client';
import { MagnifyingGlass, Command } from '@phosphor-icons/react';

interface SearchInputProps {
    value: string;
    onChange: (value: string) => void;
}

export function SearchInput({ value, onChange }: SearchInputProps) {
    return (
        <div className="relative group max-w-2xl mx-auto">
            <div className="absolute inset-0 bg-accent/20 rounded-xl blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-500" />
            <div className="relative flex items-center bg-surface/80 backdrop-blur-md border border-border rounded-xl shadow-sm focus-within:shadow-lg focus-within:border-accent/50 transition-all duration-300">
                <div className="pl-4 text-secondary group-focus-within:text-accent transition-colors">
                    <MagnifyingGlass size={20} weight="bold" />
                </div>
                <input
                    type="text"
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    placeholder="Search repositories..."
                    className="w-full bg-transparent border-none py-4 px-3 text-lg text-primary placeholder:text-secondary/50 focus:outline-none focus:ring-0"
                    autoFocus
                />
                <div className="pr-4 flex items-center gap-2 text-xs text-secondary/70">
                    <span className="flex items-center gap-1 bg-white/5 px-2 py-1 rounded border border-white/5">
                        <Command size={12} /> K
                    </span>
                </div>
            </div>
        </div>
    );
}
