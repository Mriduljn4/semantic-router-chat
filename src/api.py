import logging
from pathlib import Path
from typing import Literal
import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.agent import astream_agent
from src.guardrails import GuardrailViolation, validate_input
from src.graph import run_query_async
from src.llm import LLMProviderError
from src.router import route as decide_route

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
        return validate_input(query)


class QueryResponse(BaseModel):
    answer: str
    routed_agent: Literal["research", "coding", "data"]
    router_scores: dict[str, float]
    intent_classifier: Literal["groq"]
    llm_provider_used: Literal["groq", "gemini"]
    tools_used: list[str]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def chat_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        return QueryResponse.model_validate(await run_query_async(request.query))
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
    except GuardrailViolation as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (RuntimeError, ValueError) as error:
        logger.exception("Query processing failed.")
        raise HTTPException(status_code=503, detail="Query processing is temporarily unavailable.") from error


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    def event(event_name: str, payload: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"

    async def generate():
        yield event("status", {"message": "Routing your question…"})
        try:
            decision = await asyncio.to_thread(decide_route, request.query)
            yield event("status", {"message": f"{decision.routed_agent.title()} specialist is preparing an answer…"})
            async for stream_event in astream_agent(decision.routed_agent, request.query):
                if stream_event["type"] == "start":
                    yield event(
                        "answer_start",
                        {
                            "routed_agent": decision.routed_agent,
                            "router_scores": decision.router_scores,
                            "intent_classifier": decision.classifier_used,
                            "llm_provider_used": stream_event["provider"],
                            "tools_used": stream_event["tools_used"],
                        },
                    )
                elif stream_event["type"] == "status":
                    yield event("status", {"message": stream_event["message"]})
                elif stream_event["type"] == "token":
                    yield event("answer_chunk", {"text": stream_event["text"]})
            yield event("answer_complete", {})
        except LLMProviderError as error:
            logger.exception("All LLM providers failed to generate a response.")
            yield event(
                "error",
                {
                    "message": "Language model service is unavailable.",
                    "reason": error.reason,
                    "attempts": error.attempts,
                },
            )
        except GuardrailViolation as error:
            yield event("error", {"message": str(error), "reason": "input_guardrail"})
        except (RuntimeError, ValueError):
            logger.exception("Query processing failed.")
            yield event("error", {"message": "Query processing is temporarily unavailable."})

    return StreamingResponse(generate(), media_type="text/event-stream")
