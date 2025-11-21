# Ticket 05: "Ask AI" Chat Interface

**Goal**: Implement the RAG-powered chat interface where users can ask questions about their repositories.

## Context
This is the "Wow" feature. It shouldn't just be a text box; it should feel like a conversation with the codebase. It needs to show *citations* (which repos were used).

## Requirements

### 1. Chat UI (`components/ChatPanel.tsx`)
- [ ] **Location**: Slide-over panel from the right, or a dedicated mode.
- [ ] **Input**: "Ask a question..." with support for `CMD+Enter` to submit.
- [ ] **Messages**:
  - User message: Right aligned, simple bubble.
  - AI message: Left aligned, markdown support, streaming text.
- [ ] **Citations**:
  - Below the answer, show "Sources: [Repo A], [Repo B]".
  - Hovering a citation highlights the relevant part of the answer (advanced) or just links to the repo.

### 2. Streaming Hook (`hooks/useChat.ts`)
- [ ] Use `ai/react` (Vercel AI SDK) or a custom `fetch` with `ReadableStream`.
- [ ] Handle `onFinish` to save the conversation to local history (optional for MVP).

### 3. Backend Endpoint (`src/api/ask.py`)
- [ ] `POST /api/v1/ask`:
  - Input: `{ "query": "...", "history": [...] }`.
  - Logic:
    1. Embed query.
    2. Retrieve top 5 chunks from Qdrant.
    3. Rerank (optional).
    4. Construct prompt with context.
    5. Stream response from OpenRouter.

## Implementation Snippets

**`ui/components/ChatPanel.tsx`**:
```tsx
import { PaperPlaneRight, Sparkle } from '@phosphor-icons/react';
import ReactMarkdown from 'react-markdown';

export function ChatPanel() {
  const { messages, input, handleInputChange, handleSubmit } = useChat(); // Vercel AI SDK

  return (
    <div className="flex flex-col h-full bg-surface border-l border-border w-[400px]">
      <div className="p-4 border-b border-border flex items-center gap-2">
        <Sparkle className="text-accent" />
        <h2 className="font-semibold">Ask AI</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map(m => (
          <div key={m.id} className={clsx(
            "p-3 rounded-lg text-sm",
            m.role === 'user' ? "bg-accent text-white self-end" : "bg-white/5 text-secondary self-start"
          )}>
            <ReactMarkdown>{m.content}</ReactMarkdown>
            {/* Render citations if present in m.data */}
          </div>
        ))}
      </div>
      
      <form onSubmit={handleSubmit} className="p-4 border-t border-border">
        <div className="relative">
          <input
            value={input}
            onChange={handleInputChange}
            placeholder="Ask about your repos..."
            className="w-full bg-background border border-border rounded-md pl-3 pr-10 py-2 focus:ring-1 focus:ring-accent outline-none"
          />
          <button type="submit" className="absolute right-2 top-2 text-secondary hover:text-primary">
            <PaperPlaneRight size={20} />
          </button>
        </div>
      </form>
    </div>
  );
}
```

**`src/api/ask.py` (Streaming)**:
```python
from fastapi.responses import StreamingResponse

@router.post("/ask")
async def ask(req: AskRequest):
    async def generate():
        context = retrieve_context(req.query)
        async for chunk in llm.stream_chat(req.query, context):
            yield chunk
            
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Acceptance Criteria
- Chat panel opens/closes.
- Streaming text works (no full-buffer wait).
- Markdown renders correctly.
- Sources are listed at the bottom of the answer.
