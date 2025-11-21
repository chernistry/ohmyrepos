# Ticket 03: Next.js UI MVP

**Goal**: Build the initial web interface for Oh My Repos using Next.js 14 (App Router).

## Context
The UI will be the primary entry point for users, replacing the Streamlit app. It must be fast, responsive, and visually appealing.

## Requirements
- [ ] **Project Initialization**:
    - Initialize a new Next.js app in `ui/` (or `src/ui` if preferred, but `ui/` is cleaner for monorepo-like structure).
    - Use TypeScript, Tailwind CSS, ESLint.
- [ ] **Core Components**:
    - `SearchBar`: Main input with debounce.
    - `RepoCard`: Display repo details (name, stars, language, summary, tags).
    - `SearchResults`: List of `RepoCard`s.
    - `FilterPanel`: Facets for Language, Stars, etc.
- [ ] **Pages**:
    - `app/page.tsx`: Main search page.
    - `app/repo/[id]/page.tsx`: Detailed view (optional for MVP, but good for "Ask" context).
- [ ] **API Integration**:
    - Create a typed API client (using `fetch` or `tanstack-query`).
    - Connect to the FastAPI backend (use `NEXT_PUBLIC_API_BASE_URL`).
- [ ] **Styling**:
    - Use a clean, modern aesthetic (dark mode support recommended).
    - Ensure mobile responsiveness.

## Acceptance Criteria
- Next.js app builds and runs locally (`pnpm dev`).
- User can type a query and see results from the API.
- Clicking a result opens the repo URL or detail view.
- Filters (e.g., Language) work and update the search results.
