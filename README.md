# Agents.ai & Semantic Router

A lean FastAPI multi-agent service that routes questions to Research, Coding, or Data experts. Routing uses local sentence-transformer embeddings against ChromaDB capability profiles. The Research agent performs retrieval-augmented generation (RAG) over a small local document collection. Groq is the primary LLM provider and Gemini is the automatic fallback.

## Architecture

`POST /query` invokes a two-node LangGraph: `route` embeds the query and selects the highest mean capability similarity; `run_agent` executes the selected LangChain v1 specialist agent. Research prompts include retrieved context, and the LangChain agent uses Groq with a Gemini fallback. LangGraph automatically traces runs when LangSmith environment variables are configured.

## Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

3. Copy `.env.example` to `.env` and add free Groq and Gemini API keys.
4. Seed the local Chroma collections:

```bash
python -m scripts.seed
```

5. Start the application:

```bash
uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000 for the chat UI. The API remains available at `/query`, with interactive documentation at `/docs`.

## Query example

```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"query\": \"What is retrieval-augmented generation?\"}"
```

Example response:

```json
{
  "answer": "...",
  "routed_agent": "research",
  "router_scores": {"research": 0.82, "coding": 0.15, "data": 0.09},
  "llm_provider_used": "groq"
}
```

Use `GET /health` for a lightweight service check.

## Tests

Run all tests with:

```bash
pytest tests/
```

The router, fallback, and API tests make no LLM API calls. The DeepEval RAG quality check is an external integration test and is skipped by default. Run it only when Groq credentials and verified network access are available by setting both `GROQ_API_KEY` and `RUN_INTEGRATION_TESTS=true`.

## Local storage

ChromaDB persists its local data in `chroma_data/` using SQLite. This setup is appropriate for local development and read-heavy single-process use. Use a hosted/vector-database deployment before scaling to concurrent production writers.

## Deploy free on Render

This repository includes [render.yaml](render.yaml), which creates a free Render web service and seeds the Chroma data during every deployment build. It uses the lightweight deterministic `hashing` embedding backend and [requirements-render.txt](requirements-render.txt) on Render to stay within the free instance's 512 MB memory limit. Local development continues to use the higher-quality sentence-transformer backend by default.

1. Sign in at [Render](https://render.com/) using GitHub.
2. Select **New +** → **Blueprint**, then select the `Mriduljn4/semantic-router-chat` repository.
3. Review the `semantic-router-chat` service and click **Apply**.
4. In the service **Environment** settings, add `GROQ_API_KEY`. Optionally add `GEMINI_API_KEY` for fallback.
5. Redeploy the service after saving the environment variables.

Render will provide the public URL after deployment. Free services sleep when idle and may take a short time to wake up for the next request.

## Next steps

A next iteration could add a confidence threshold that sends uncertain requests to a general fallback, expand the small labeled routing set into a measured evaluation dataset, and deploy the same FastAPI service to a free host such as Render. These ideas are intentionally not implemented in this lean version.
