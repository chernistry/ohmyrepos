'use client';
import { useState } from 'react';
import { MagnifyingGlass, Stack, Gear, Plus } from '@phosphor-icons/react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { clsx } from 'clsx';
import { SyncStatus } from './SyncStatus';
import { AddRepoModal } from './AddRepoModal';

const NAV_ITEMS = [
    { label: 'Discover', icon: MagnifyingGlass, href: '/' },
    { label: 'Collections', icon: Stack, href: '/collections' },
    { label: 'Settings', icon: Gear, href: '/settings' },
];

export function Sidebar() {
    const pathname = usePathname();
    const [isModalOpen, setIsModalOpen] = useState(false);

    return (
        <>
            <aside className="w-64 h-full border-r border-border bg-surface/50 backdrop-blur-xl flex flex-col p-4">
                <div className="mb-8 px-2 flex justify-between items-center">
                    <h1 className="font-bold text-xl tracking-tight text-primary">OhMyRepos</h1>
                    <button
                        onClick={() => setIsModalOpen(true)}
                        className="p-1 rounded-md text-secondary hover:text-primary hover:bg-white/10 transition-colors"
                        title="Add Repository"
                    >
                        <Plus size={18} weight="bold" />
                    </button>
                </div>

                <nav className="flex-1 space-y-1">
                    {NAV_ITEMS.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={clsx(
                                    "flex items-center gap-3 px-3 py-2 rounded-md transition-all duration-200 group",
                                    isActive
                                        ? "bg-white/10 text-white shadow-sm"
                                        : "text-secondary hover:text-primary hover:bg-white/5"
                                )}
                            >
                                <item.icon size={18} weight={isActive ? "fill" : "regular"} />
                                <span className="text-sm font-medium">{item.label}</span>
                            </Link>
                        );
                    })}
                </nav>

                <SyncStatus />
            </aside>

            <AddRepoModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
        </>
    );
}
