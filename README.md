# minime — Mini AI assistant with Electron + React + Express + OpenAI

A local Electron app with React frontend, Express backend, and OpenAI integration for chat, embeddings, and vector-based retrieval.

## Project structure

```
minime/
├── client/          # Vite + React + Electron
├── server/          # Express backend (LLM, embeddings, vectors, OS actions)
├── package.json     # Root scripts (dev, build)
└── README.md
```

## Quick start

### Prerequisites
- Node.js 18+
- Git (installed; see below if missing)
- (Optional) OpenAI API key for LLM/embeddings

### 1. Install & run

From the project root:

```bash
npm run dev
```

This runs:
- **Server** on `http://localhost:8000` (Express + in-memory vector store)
- **Client** on `http://localhost:5173` (Vite dev server) + Electron desktop window

### 2. Configure (optional)

Create `server/.env` from the example:

```bash
cp server/.env.example server/.env
```

Edit `server/.env`:

```env
# Optional: your OpenAI API key (if missing, server echoes messages)
OPENAI_API_KEY=sk-...

# Required if you call protected endpoints (e.g., /api/chat without ROOT_API_KEY → 401)
ROOT_API_KEY=dev-secret

# Optional: model overrides
CHAT_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small

# Server port
PORT=8000

# Optional: enable Chroma persistence (requires Chroma server or provider setup)
ENABLE_CHROMA=false
```

### 3. API endpoints

**Base URL:** `http://localhost:8000`

All protected endpoints require `x-api-key: <ROOT_API_KEY>` header (except `/health`).

#### Health
```
GET /health
→ { ok: true, openai: <boolean> }
```

#### Chat (with RAG + LLM)
```
POST /api/chat
Header: x-api-key: dev-secret
Body: { "message": "Hello" }
→ { "reply": "...", "stored": { id, text, embedding, meta } }
```

#### Add vector to store
```
POST /api/vector
Header: x-api-key: dev-secret
Body: { "text": "Document text", "meta": {...} }
→ { "ok": true, "item": { id, text, embedding, meta } }
```

#### OS actions (whitelisted)
```
POST /api/action
Header: x-api-key: dev-secret
Body: { "action": "open-folder", "args": { "path": "C:/Users/..." } }
     or { "action": "open-chrome-new-window", "args": { "url": "https://..." } }
→ { "ok": true }
```

## Features

- **LLM chat**: calls OpenAI API (gpt-3.5-turbo or configurable model); falls back to echo if no API key.
- **Embeddings**: uses text-embedding-3-small to embed messages and store documents.
- **In-memory vector DB**: simple in-memory store with cosine similarity retrieval (top-k).
- **RAG**: retrieves top-3 similar documents as context for LLM prompts.
- **OS actions**: platform-specific (Windows/Mac/Linux) folder opening and Chrome launches.
- **API key auth**: simple middleware checks `x-api-key` header (optional if no ROOT_API_KEY set).
- **Electron desktop app**: Vite dev server + Electron wrapper for desktop UX.

## Optional features (not yet enabled)

- **Chroma persistence**: set `ENABLE_CHROMA=true` and run a Chroma server or configure provider. Replaces in-memory store with persistent vector DB.
- **Streaming**: implement SSE or WebSocket for progressive LLM responses.
- **Tests & CI**: add Jest/Mocha tests and GitHub Actions workflows.

## Development

### Building the Electron app
```bash
npm run build
```

This builds:
- Vite bundle → `client/dist/`
- Electron app → `.out/` or platform-specific installer

### Running just the server
```bash
cd server
npm run dev
```

### Running just the client dev server
```bash
cd client
npm run dev
```

## Troubleshooting

**Git not found**: install from https://git-scm.com/download/win or run `winget install Git.Git`.

**npm install fails**: ensure Node.js 18+ is installed; check `npm --version`.

**Server won't start**: 
- Check if port 8000 is already in use: `netstat -ano | findstr :8000` (Windows).
- Ensure `.env` is correct (if present).

**Electron window doesn't open**: 
- Check client dev server is running on port 5173.
- Allow a few seconds for Vite to compile.

**LLM returns "Echo (no OPENAI_API_KEY)"**: 
- Set `OPENAI_API_KEY` in `server/.env` to enable real LLM calls.

## License

ISC (see individual package.json files).

---

**Next steps:**
- Add your OpenAI API key to `server/.env`.
- Test the chat endpoint via the client UI or `curl`.
- Optionally enable Chroma for persistent vectors (requires Chroma server setup).
- Add tests and CI workflows as needed.
