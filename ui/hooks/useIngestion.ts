import { useState, useEffect } from 'react';
import { toast } from 'sonner';

export interface SyncStatusData {
    status: 'idle' | 'running' | 'error';
    progress: number;
    total: number;
    lastSyncedAt: string | null;
}

export function useIngestion() {
    const [status, setStatus] = useState<SyncStatusData>({
        status: 'idle',
        progress: 0,
        total: 0,
        lastSyncedAt: null, // In real app, fetch from API
    });

    const triggerSync = async () => {
        setStatus(prev => ({ ...prev, status: 'running', progress: 0 }));
        toast.info('Sync started...');

        try {
            // Mock API call
            await new Promise(resolve => setTimeout(resolve, 1000));

            // Mock progress
            let p = 0;
            const interval = setInterval(() => {
                p += 10;
                setStatus(prev => ({ ...prev, progress: p, total: 100 }));
                if (p >= 100) {
                    clearInterval(interval);
                    setStatus(prev => ({ ...prev, status: 'idle', lastSyncedAt: new Date().toISOString() }));
                    toast.success('Sync completed');
                }
            }, 500);

        } catch (e) {
            toast.error('Sync failed');
            setStatus(prev => ({ ...prev, status: 'error' }));
        }
    };

    const addRepo = async (url: string) => {
        try {
            // Mock API call
            await new Promise(resolve => setTimeout(resolve, 1000));
            toast.success(`Added ${url}`);
            return true;
        } catch (e) {
            toast.error('Failed to add repo');
            return false;
        }
    };

    return { status, triggerSync, addRepo };
}
