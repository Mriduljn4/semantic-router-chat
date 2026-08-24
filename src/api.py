import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.graph import run_query
from src.llm import LLMProviderError

logger = logging.getLogger(__name__)

app = FastAPI(title="Agents.ai & Semantic Router")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("Query cannot be blank.")
        return query


class QueryResponse(BaseModel):
    answer: str
    routed_agent: Literal["research", "coding", "data"]
    router_scores: dict[str, float]
    llm_provider_used: Literal["groq", "gemini"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def chat_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return QueryResponse.model_validate(run_query(request.query))
    except LLMProviderError as error:
        logger.exception("All LLM providers failed to generate a response.")
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Language model service is unavailable.",
                "reason": error.reason,
                "attempts": error.attempts,
            },
        ) from error
    except (RuntimeError, ValueError) as error:
        logger.exception("Query processing failed.")
        raise HTTPException(status_code=503, detail="Query processing is temporarily unavailable.") from error
