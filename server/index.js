// server/index.js
const express = require('express')
const cors = require('cors')
const { spawn } = require('child_process')
const { GoogleGenerativeAI } = require('@google/generative-ai')
// chromadb optional: require if present; we'll fall back to in-memory store if not
let Chroma = null
try { Chroma = require('chromadb') } catch (e) { Chroma = null }
// load env if available (optional)
try { require('dotenv').config() } catch (e) {}

console.log('ENABLE_CHROMA:', process.env.ENABLE_CHROMA, 'Chroma available:', !!Chroma)

const app = express()
app.use(cors())
app.use(express.json())

// --- API key middleware ---
const ROOT_API_KEY = process.env.ROOT_API_KEY || null
function apiKeyMiddleware(req, res, next) {
  // allow health checks without a key
  if (req.path === '/health') return next()
  // if no ROOT_API_KEY configured, allow all (convenient for local dev)
  if (!ROOT_API_KEY) return next()
  const key = req.headers['x-api-key'] || req.query.api_key
  if (!key || key !== ROOT_API_KEY) return res.status(401).json({ error: 'missing or invalid API key' })
  return next()
}
app.use(apiKeyMiddleware)

// Gemini client (Google) - used for chat generation
const genai = process.env.OPENAI_API_KEY ? new GoogleGenerativeAI(process.env.OPENAI_API_KEY) : null

// Optional OpenAI SDK usage
let OpenAIPkg = null
let openaiChatClient = null
let openaiEmbeddingsClient = null
try {
  OpenAIPkg = require('openai')
} catch (e) {
  OpenAIPkg = null
}

// Initialize OpenAI chat client if OPENAI_API_KEY is provided
if (OpenAIPkg && process.env.OPENAI_API_KEY) {
  try {
    openaiChatClient = new OpenAIPkg({ apiKey: process.env.OPENAI_API_KEY })
    console.info('OpenAI chat client initialized')
  } catch (e) {
    openaiChatClient = null
  }
}

// Initialize OpenAI embeddings client if OPENAI_EMBEDDING_KEY is provided
if (OpenAIPkg && process.env.OPENAI_EMBEDDING_KEY) {
  try {
    openaiEmbeddingsClient = new OpenAIPkg({ apiKey: process.env.OPENAI_EMBEDDING_KEY })
    console.info('OpenAI embeddings client initialized')
  } catch (e) {
    openaiEmbeddingsClient = null
  }
}

// --- Vector store implementation ---
// Default: simple in-memory vector store. If chromadb is available we'll use it;
// otherwise we fall back to the in-memory store.
const vectorStore = {
  type: 'memory',
  vectors: [] // { id, text, embedding, meta }
}

let chromaClient = null
let chromaCollection = null
// Only attempt to initialize Chroma when explicitly enabled via ENABLE_CHROMA env var.
// This avoids startup failures when Chroma server is not running or client is not configured.
// To enable Chroma: set ENABLE_CHROMA=true and configure Chroma Cloud or local server via env vars.
if (process.env.ENABLE_CHROMA === 'true' && Chroma) {
  ;(async () => {
    try {
      // Use Chroma Cloud if credentials are provided
      if (process.env.CHROMA_CLOUD_API_KEY && process.env.CHROMA_TENANT) {
        chromaClient = new Chroma.CloudClient({
          apiKey: process.env.CHROMA_CLOUD_API_KEY,
          tenant: process.env.CHROMA_TENANT,
          database: process.env.CHROMA_DATABASE || 'minime'
        })
        console.info('Connected to Chroma Cloud')
      } else {
        // Fall back to local Chroma server
        if (typeof Chroma === 'function') chromaClient = new Chroma()
        else if (Chroma.ChromaClient) chromaClient = new Chroma.ChromaClient()
        else if (Chroma.Client) chromaClient = new Chroma.Client()
        else chromaClient = new Chroma()
        console.info('Connected to local Chroma server')
      }

      // Try to get or create a collection named 'minime'
      if (chromaClient) {
        try {
          if (chromaClient.getCollection) {
            // First try to get existing collection
            chromaCollection = await chromaClient.getCollection({ name: 'minime' })
            console.info('Got existing Chroma collection: minime')
          }
        } catch (e) {
          // Collection doesn't exist; try to create it
          try {
            if (chromaClient.createCollection) {
              chromaCollection = await chromaClient.createCollection({ name: 'minime' })
              console.info('Created Chroma collection: minime')
            }
          } catch (e2) {
            console.warn('Could not create collection', e2.message)
            chromaCollection = null
          }
        }
      }
      if (chromaCollection) vectorStore.type = 'chroma'
    } catch (e) {
      console.warn('chroma initialization failed, falling back to memory store', e.message)
      chromaClient = null
      chromaCollection = null
    }
  })()
} else if (Chroma && process.env.ENABLE_CHROMA !== 'true') {
  console.info('Chroma module present but disabled. Set ENABLE_CHROMA=true to enable Chroma integration.')
}

// helper: embeddings via OpenAI (optional)
// Set `OPENAI_EMBEDDING_KEY` in `.env` to enable embedding-based RAG
async function embedText(text) {
  if (!openaiClient) return null
  try {
    const resp = await openaiClient.embeddings.create({
      model: process.env.EMBEDDING_MODEL || 'text-embedding-3-small',
      input: text
    })
    return resp.data && resp.data[0] && resp.data[0].embedding
  } catch (err) {
    console.error('embedding error', err && err.message ? err.message : err)
    return null
  }
}

// helper: cosine similarity (robust to nulls)
function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return -1
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

// Add vector to store (uses chroma when available)
async function addVector({ id, text, meta }) {
  const emb = await embedText(text)
  const item = { id: id || (vectorStore.vectors.length + 1).toString(), text, embedding: emb, meta: meta || {} }

  // Try Chroma first (but since we may not have embeddings for Gemini, store documents/metadata)
  if (chromaCollection) {
    try {
      // prefer .add if available
      if (typeof chromaCollection.add === 'function') {
        await chromaCollection.add({ ids: [item.id], documents: [text], metadatas: [item.meta || {}], embeddings: item.embedding ? [item.embedding] : undefined })
        return item
      }

      // some clients use upsert or insert
      if (typeof chromaCollection.upsert === 'function') {
        await chromaCollection.upsert({ ids: [item.id], documents: [text], metadatas: [item.meta || {}], embeddings: item.embedding ? [item.embedding] : undefined })
        return item
      }

      // last resort: try calling client-level API
      if (chromaClient && typeof chromaClient.add === 'function') {
        await chromaClient.add({ ids: [item.id], documents: [text], metadatas: [item.meta || {}] })
        return item
      }
    } catch (e) {
      console.warn('chroma add/upsert failed, falling back to memory store', e && e.message ? e.message : e)
    }
  }

  // Fallback to in-memory
  vectorStore.vectors.push(item)
  return item
}

// Query top-k (uses chroma when available; note: Gemini has no embeddings so this falls back to memory)
async function queryVectors(embedding, k = 3) {
  if (!embedding) return []

  // If Chroma is available and embeddings are present, attempt a Chroma query.
  if (chromaCollection) {
    try {
      if (typeof chromaCollection.query === 'function') {
        const q = await chromaCollection.query({ query_embeddings: [embedding], n_results: k })
        const docs = (q.documents && q.documents[0]) || []
        const ids = (q.ids && q.ids[0]) || []
        const distances = (q.distances && q.distances[0]) || []
        const out = []
        for (let i = 0; i < docs.length; i++) out.push({ id: ids[i] || (i + 1).toString(), text: docs[i], score: distances[i] })
        return out
      }

      if (chromaClient && typeof chromaClient.query === 'function') {
        const q = await chromaClient.query({ query_embeddings: [embedding], n_results: k })
        const docs = (q.documents && q.documents[0]) || []
        const ids = (q.ids && q.ids[0]) || []
        const distances = (q.distances && q.distances[0]) || []
        const out = []
        for (let i = 0; i < docs.length; i++) out.push({ id: ids[i] || (i + 1).toString(), text: docs[i], score: distances[i] })
        return out
      }
    } catch (e) {
      console.warn('chroma query failed, falling back to memory', e && e.message ? e.message : e)
    }
  }

  // Fallback: in-memory similarity scoring
  const scored = vectorStore.vectors.map(v => ({ ...v, score: cosine(embedding, v.embedding) }))
  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, k)
}

// Health
app.get('/health', (req, res) => res.json({ ok: true, gemini: !!genai }))

// POST /api/vector - add a document to the vector DB
app.post('/api/vector', async (req, res) => {
  const { id, text, meta } = req.body || {}
  if (!text) return res.status(400).json({ error: 'text required' })
  try {
    const item = await addVector({ id, text, meta })
    return res.json({ ok: true, item })
  } catch (err) {
    console.error('addVector error', err)
    return res.status(500).json({ error: 'internal error' })
  }
})

// Admin: list vectors (supports chroma and memory fallback)
app.get('/api/vector/list', async (req, res) => {
  try {
    if (chromaCollection) {
      try {
        if (typeof chromaCollection.get === 'function') {
          const g = await chromaCollection.get()
          // normalize shapes: some clients return arrays of arrays
          const ids = (g.ids && (Array.isArray(g.ids[0]) ? g.ids[0] : g.ids)) || []
          const documents = (g.documents && (Array.isArray(g.documents[0]) ? g.documents[0] : g.documents)) || []
          const metadatas = (g.metadatas && (Array.isArray(g.metadatas[0]) ? g.metadatas[0] : g.metadatas)) || []
          const embeddings = (g.embeddings && (Array.isArray(g.embeddings[0]) ? g.embeddings[0] : g.embeddings)) || []
          const out = []
          const max = Math.max(ids.length, documents.length)
          for (let i = 0; i < max; i++) out.push({ id: ids[i] || null, text: documents[i] || null, meta: metadatas[i] || null, embedding: embeddings[i] || null })
          return res.json({ ok: true, items: out })
        }
      } catch (e) {
        console.warn('chroma get failed', e && e.message ? e.message : e)
      }
    }

    // fallback to in-memory
    return res.json({ ok: true, items: vectorStore.vectors })
  } catch (err) {
    console.error('list vectors error', err)
    return res.status(500).json({ error: 'internal error' })
  }
})

// Admin: clear vectors (supports chroma and memory fallback)
app.post('/api/vector/clear', async (req, res) => {
  try {
    if (chromaCollection) {
      try {
        if (typeof chromaCollection.delete === 'function') {
          // some clients implement delete
          await chromaCollection.delete()
          return res.json({ ok: true })
        }
        if (typeof chromaCollection.clear === 'function') {
          await chromaCollection.clear()
          return res.json({ ok: true })
        }
        if (chromaClient && typeof chromaClient.deleteCollection === 'function') {
          await chromaClient.deleteCollection({ name: 'minime' })
          // try to recreate empty collection
          if (typeof chromaClient.createCollection === 'function') chromaCollection = await chromaClient.createCollection({ name: 'minime' })
          return res.json({ ok: true })
        }
      } catch (e) {
        console.warn('chroma clear failed', e && e.message ? e.message : e)
      }
    }

    // fallback to in-memory clear
    vectorStore.vectors = []
    return res.json({ ok: true })
  } catch (err) {
    console.error('clear vectors error', err)
    return res.status(500).json({ error: 'internal error' })
  }
})

// Chat endpoint with Gemini
app.post('/api/chat', async (req, res) => {
  const userMessage = req.body && req.body.message
  if (!userMessage) return res.status(400).json({ error: 'message required' })
  try {
    // Gemini doesn't have embeddings, so skip RAG
    const topDocs = ''
    const prompt = `You are a helpful assistant.\nUser: ${userMessage}\nAnswer:`

    let reply = ''
    if (genai) {
      try {
        const model = genai.getGenerativeModel({ model: process.env.CHAT_MODEL || 'gemini-1.5-flash' })
        // Generate content without timeout first to test
        const result = await model.generateContent(prompt)
        if (result && result.response) {
          reply = result.response.text()
        } else {
          reply = 'Received empty response from Gemini'
        }
      } catch (err) {
        console.error('gemini error', err.message || err)
        reply = `Error: Gemini failed: ${err.message}`
      }
    } else {
      // local fallback
      reply = `Echo (no API key): ${userMessage}`
    }

    // store user message as memory
    try {
      const stored = await addVector({ text: userMessage })
      return res.json({ reply, stored })
    } catch (storeErr) {
      console.error('storage error', storeErr.message)
      return res.json({ reply, stored: null })
    }
  } catch (err) {
    console.error('chat handler error', err.message || err)
    return res.status(500).json({ error: 'internal error: ' + (err.message || 'unknown') })
  }
})

// OS actions endpoint (whitelisted actions)
const ACTION_WHITELIST = {
  'open-folder': true,
  'open-chrome-new-window': true
}

app.post('/api/action', async (req, res) => {
  const { action, args } = req.body || {}
  if (!ACTION_WHITELIST[action]) return res.status(403).json({ error: 'action not allowed' })
  try {
    if (action === 'open-folder') {
      const folder = args && args.path
      if (!folder) return res.status(400).json({ error: 'path required' })
      if (process.platform === 'win32') {
        spawn('explorer', [folder], { detached: true })
      } else if (process.platform === 'darwin') {
        spawn('open', [folder], { detached: true })
      } else {
        spawn('xdg-open', [folder], { detached: true })
      }
      return res.json({ ok: true })
    } else if (action === 'open-chrome-new-window') {
      const url = (args && args.url) || 'about:blank'
      if (process.platform === 'win32') {
        spawn('cmd', ['/c', 'start', 'chrome', url], { detached: true })
      } else if (process.platform === 'darwin') {
        spawn('open', ['-a', 'Google Chrome', url], { detached: true })
      } else {
        spawn('google-chrome', [url], { detached: true })
      }
      return res.json({ ok: true })
    }
  } catch (e) {
    console.error(e)
    return res.status(500).json({ error: e.message })
  }
})

const port = process.env.PORT || 8000
const server = app.listen(port, () => console.log('Server listening on', port))

// Graceful error handling
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason)
})

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err)
  process.exit(1)
})
