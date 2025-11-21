# Ticket 03: Ingestion & Sync UI

**Goal**: Implement the UI for adding repositories and syncing them (incrementally).

## Context
Users need to add repos (by URL or "Sync All Starred"). The UI should show progress and handle the "incremental" nature (only updating what changed).

## Requirements

### 1. "Add Repo" Modal (`components/AddRepoModal.tsx`)
- [ ] **Trigger**: "+" button in Sidebar or Command Palette.
- [ ] **Input**: Text field for `owner/repo` or full URL.
- [ ] **Action**: POST to `/api/v1/ingest`.
- [ ] **Feedback**: Spinner during check, Success toast on completion.

### 2. Sync Status & Control (`components/SyncStatus.tsx`)
- [ ] **Location**: Bottom of Sidebar.
- [ ] **States**:
  - `Idle`: "Last synced 2h ago". Hover shows "Sync Now" button.
  - `Syncing`: Progress bar (or spinning icon) with "Checking 12/50...".
  - `Error`: Red dot with tooltip.
- [ ] **Logic**:
  - On mount, fetch sync status.
  - "Sync Now" triggers `/api/v1/sync`.
  - Poll `/api/v1/sync/status` every 2s during sync (or use SSE if ambitious, polling is fine for MVP).

### 3. Hook: `useIngestion`
- [ ] Manage state for `isSyncing`, `progress`, `lastSyncedAt`.
- [ ] Expose `addRepo(url)` and `triggerSync()`.

## Implementation Snippets

**`ui/hooks/useIngestion.ts`**:
```ts
import { useState } from 'react';
import { toast } from 'sonner'; // Recommended toast lib

export function useIngestion() {
  const [isSyncing, setIsSyncing] = useState(false);
  
  const triggerSync = async () => {
    setIsSyncing(true);
    try {
      // Trigger the background job
      await fetch('/api/v1/sync', { method: 'POST' });
      toast.info('Sync started...');
      
      // Start polling logic here...
    } catch (e) {
      toast.error('Sync failed');
      setIsSyncing(false);
    }
  };
  
  return { isSyncing, triggerSync };
}
```

**`ui/components/SyncStatus.tsx`**:
```tsx
import { ArrowsClockwise, CheckCircle } from '@phosphor-icons/react';
import { useIngestion } from '@/hooks/useIngestion';

export function SyncStatus() {
  const { isSyncing, triggerSync } = useIngestion();
  
  return (
    <div className="px-4 py-3 border-t border-border">
      <button 
        onClick={triggerSync}
        disabled={isSyncing}
        className="flex items-center gap-3 w-full text-sm text-secondary hover:text-primary transition-colors disabled:opacity-50"
      >
        <ArrowsClockwise 
          size={16} 
          className={isSyncing ? "animate-spin" : ""} 
        />
        <div className="flex flex-col items-start">
          <span className="font-medium">
            {isSyncing ? 'Syncing...' : 'Up to date'}
          </span>
          {!isSyncing && (
            <span className="text-xs text-secondary/70">Last: 2m ago</span>
          )}
        </div>
      </button>
    </div>
  );
}
```

## Acceptance Criteria
- "Sync Now" button triggers the API.
- UI reflects "Syncing" state immediately.
- "Add Repo" works for a single URL.
