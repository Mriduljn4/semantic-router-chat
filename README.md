# Semantic Router Chat

A FastAPI chat app that routes questions to Research, Coding, or Data specialists.

## How it works

1. Guardrails validate the user query.
2. Groq structured output selects the intent: Research, Coding, or Data.
3. ChromaDB calculates semantic similarity scores for visibility only; it does not override Groq's intent selection.
4. The specialist generates an answer. Research retrieves local RAG sources and can use public web search for timely information.

## Run locally

1. Copy `.env.example` to `.env` and add the required API key.
2. Install dependencies:

```bash
python -m pip install -e .
```

3. Seed the local data:

```bash
python -m scripts.seed
```

4. Start the app:

```bash
uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000.

## API

- `GET /health` — service health check
- `POST /query` — complete response
- `POST /query/stream` — streamed chat response
- `GET /docs` — API documentation

Example request:

```json
{"query": "What is retrieval-augmented generation?"}
```

## Deployment

`render.yaml` contains the free Render deployment configuration. Add `GEMINI_API_KEY` in the Render environment settings before deploying.
