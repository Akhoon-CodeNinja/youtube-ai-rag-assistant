# YouTube AI RAG Assistant — React Frontend

## Quick Setup
```bash
npm install
npm run dev
```

## Backend API Endpoints
Toggle **Live Backend API** in the top-right header (`http://localhost:8000/api`) to connect to your Python server:
- `POST /api/index` -> `{ "url": "..." }` returns `{ "title", "channel", "duration", "chunks" }`
- `POST /api/ask` -> `{ "question", "video_url", "language" }` returns `{ "answer", "verified", "confidence", "latency", "contexts": [{ "timestamp", "score", "text", "highlight" }] }`
