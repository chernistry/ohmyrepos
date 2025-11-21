# Oh My Repos UI

Next.js 16 frontend for Oh My Repos semantic search.

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Configuration

Create `.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Features

- **Search Bar**: Real-time search with debouncing
- **Repo Cards**: Display repository details with tags, stars, and language
- **Responsive Design**: Mobile-friendly Tailwind CSS styling
- **Error Handling**: User-friendly error messages

## Components

- `SearchBar` - Search input with submit handling
- `RepoCard` - Individual repository result card
- `SearchResults` - List of search results

## API Integration

The UI connects to the FastAPI backend at `NEXT_PUBLIC_API_BASE_URL`.

Endpoint: `POST /api/v1/search`

## Tech Stack

- Next.js 16 (App Router)
- React 19
- TypeScript
- Tailwind CSS 4
- ESLint
