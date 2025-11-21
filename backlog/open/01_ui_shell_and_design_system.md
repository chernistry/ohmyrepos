# Ticket 01: UI Shell & Design System

**Goal**: Establish the "Premium" look and feel using Next.js 14, Tailwind, and Phosphor Icons.

## Context
The user wants a "cardinal redesign" — think Vercel, Linear, Raycast. Dark mode by default, crisp typography, subtle borders, and glassmorphism.

## Tech Stack
- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Icons**: `@phosphor-icons/react`
- **Fonts**: `Geist Sans` (or Inter) + `JetBrains Mono` for code.
- **Animation**: `framer-motion`

## Requirements

### 1. Setup & Configuration
- [ ] Initialize Next.js app in `ui/`.
- [ ] Install dependencies:
  ```bash
  pnpm add @phosphor-icons/react framer-motion clsx tailwind-merge
  ```
- [ ] Configure `tailwind.config.ts` for a "premium dark" palette.
  ```ts
  // tailwind.config.ts snippet
  export default {
    theme: {
      extend: {
        colors: {
          background: '#0a0a0a', // Deep black/gray
          surface: '#111111',    // Slightly lighter
          border: '#333333',     // Subtle border
          primary: '#EDEDED',    // High contrast text
          secondary: '#A1A1A1',  // Muted text
          accent: '#3B82F6',     // Blue accent (or customizable)
        },
        fontFamily: {
          sans: ['var(--font-geist-sans)'],
          mono: ['var(--font-geist-mono)'],
        }
      }
    }
  }
  ```

### 2. Global Layout (`app/layout.tsx`)
- [ ] Create a persistent **Sidebar** navigation.
- [ ] Create a **Main Content Area** that scrolls independently.
- [ ] **Sidebar Specs**:
  - Fixed width (e.g., 240px).
  - Glass effect background (`backdrop-blur-md`).
  - Logo area at top.
  - Navigation links: "Search", "Collections", "Settings".
  - Bottom area: "Sync Status" (placeholder for now).

### 3. Component: `SidebarItem`
- [ ] Interactive state: Hover adds a subtle background `bg-white/5`.
- [ ] Active state: `text-white bg-white/10 border-l-2 border-accent`.
- [ ] Icon integration:
  ```tsx
  import { House, MagnifyingGlass, Gear } from '@phosphor-icons/react';
  
  // Usage
  <SidebarItem icon={<MagnifyingGlass size={20} />} label="Search" active />
  ```

## Implementation Snippets

**`ui/app/layout.tsx` Structure**:
```tsx
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-primary flex h-screen overflow-hidden selection:bg-accent/30">
        <Sidebar />
        <main className="flex-1 relative overflow-y-auto scrollbar-hide">
          {children}
        </main>
      </body>
    </html>
  );
}
```

**`ui/components/Sidebar.tsx`**:
```tsx
'use client';
import { MagnifyingGlass, Stack, Gear, ArrowsClockwise } from '@phosphor-icons/react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { clsx } from 'clsx';

const NAV_ITEMS = [
  { label: 'Discover', icon: MagnifyingGlass, href: '/' },
  { label: 'Collections', icon: Stack, href: '/collections' },
  { label: 'Settings', icon: Gear, href: '/settings' },
];

export function Sidebar() {
  const pathname = usePathname();
  
  return (
    <aside className="w-64 h-full border-r border-border bg-surface/50 backdrop-blur-xl flex flex-col p-4">
      <div className="mb-8 px-2">
        <h1 className="font-bold text-xl tracking-tight">OhMyRepos</h1>
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
      
      {/* Sync Status Placeholder */}
      <div className="mt-auto pt-4 border-t border-border">
        <div className="flex items-center gap-2 text-xs text-secondary px-2">
          <ArrowsClockwise size={14} className="animate-spin-slow" />
          <span>Syncing...</span>
        </div>
      </div>
    </aside>
  );
}
```

## Acceptance Criteria
- Next.js app runs.
- Dark mode is locked on.
- Sidebar is visible and responsive.
- Phosphor icons render correctly.
- Fonts are loaded.
