'use client';
import { ArrowsClockwise, CheckCircle, WarningCircle } from '@phosphor-icons/react';
import { useIngestion } from '@/hooks/useIngestion';
import { clsx } from 'clsx';

export function SyncStatus() {
    const { status, triggerSync } = useIngestion();

    return (
        <div className="px-4 py-3 border-t border-border mt-auto">
            <button
                onClick={triggerSync}
                disabled={status.status === 'running'}
                className="flex items-center gap-3 w-full text-sm text-secondary hover:text-primary transition-colors disabled:opacity-50 group"
            >
                <div className={clsx(
                    "p-1.5 rounded-md transition-colors",
                    status.status === 'running' ? "bg-accent/10 text-accent" : "bg-white/5 group-hover:bg-white/10"
                )}>
                    {status.status === 'running' ? (
                        <ArrowsClockwise size={16} className="animate-spin" />
                    ) : status.status === 'error' ? (
                        <WarningCircle size={16} className="text-red-500" />
                    ) : (
                        <ArrowsClockwise size={16} />
                    )}
                </div>

                <div className="flex flex-col items-start">
                    <span className="font-medium">
                        {status.status === 'running' ? `Syncing ${status.progress}%` : 'Up to date'}
                    </span>
                    <span className="text-xs text-secondary/70">
                        {status.status === 'running'
                            ? 'Checking repos...'
                            : status.lastSyncedAt
                                ? 'Just now'
                                : 'Sync ready'}
                    </span>
                </div>
            </button>
        </div>
    );
}
