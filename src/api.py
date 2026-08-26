import asyncio
import json
import logging
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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


ConversationAgent = Literal["general", "research", "coding", "data"]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    conversation_id: UUID = Field(default_factory=uuid4)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()

        if not query:
            raise ValueError("Query cannot be blank.")

        return validate_input(query)


class QueryResponse(BaseModel):
    answer: str
    conversation_id: UUID
    routed_agent: ConversationAgent
    router_scores: dict[str, float]
    intent_classifier: Literal["groq"]
    llm_provider_used: Literal["groq", "nvidia"]
    tools_used: list[str]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def chat_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Process a non-streaming request.

    The frontend must reuse the returned conversation_id for all later
    messages belonging to the same chat.
    """
    conversation_id = str(request.conversation_id)

    try:
        result = await run_query_async(
            query=request.query,
            conversation_id=conversation_id,
        )

        result["conversation_id"] = request.conversation_id

        return QueryResponse.model_validate(result)

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
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except (RuntimeError, ValueError) as error:
        logger.exception("Query processing failed.")

        raise HTTPException(
            status_code=503,
            detail="Query processing is temporarily unavailable.",
        ) from error


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """
    Stream an agent response as Server-Sent Events.

    The client must store conversation_id from answer_start and send the same
    ID for follow-up messages in the same conversation.
    """
    conversation_id = str(request.conversation_id)

    def event(event_name: str, payload: dict[str, object]) -> str:
        return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"

    async def generate():
        yield event(
            "status",
            {
                "message": "Routing your question…",
                "conversation_id": conversation_id,
            },
        )

        try:
            decision = await asyncio.to_thread(
                decide_route,
                request.query,
            )

            yield event(
                "status",
                {
                    "message": (
                        f"{decision.routed_agent.title()} specialist "
                        "is preparing an answer…"
                    ),
                    "conversation_id": conversation_id,
                },
            )

            async for stream_event in astream_agent(
                agent_name=decision.routed_agent,
                query=request.query,
                conversation_id=conversation_id,
            ):
                if stream_event["type"] == "start":
                    yield event(
                        "answer_start",
                        {
                            "conversation_id": conversation_id,
                            "routed_agent": decision.routed_agent,
                            "router_scores": decision.router_scores,
                            "intent_classifier": decision.classifier_used,
                            "llm_provider_used": stream_event["provider"],
                            "tools_used": stream_event["tools_used"],
                        },
                    )

                elif stream_event["type"] == "status":
                    yield event(
                        "status",
                        {
                            "message": stream_event["message"],
                            "conversation_id": conversation_id,
                        },
                    )

                elif stream_event["type"] == "token":
                    yield event(
                        "answer_chunk",
                        {
                            "text": stream_event["text"],
                            "conversation_id": conversation_id,
                        },
                    )

            yield event(
                "answer_complete",
                {
                    "conversation_id": conversation_id,
                },
            )

        except LLMProviderError as error:
            logger.exception("All LLM providers failed to generate a response.")

            yield event(
                "error",
                {
                    "message": "Language model service is unavailable.",
                    "reason": error.reason,
                    "attempts": error.attempts,
                    "conversation_id": conversation_id,
                },
            )

        except GuardrailViolation as error:
            yield event(
                "error",
                {
                    "message": str(error),
                    "reason": "input_guardrail",
                    "conversation_id": conversation_id,
                },
            )

        except (RuntimeError, ValueError):
            logger.exception("Query processing failed.")

            yield event(
                "error",
                {
                    "message": "Query processing is temporarily unavailable.",
                    "conversation_id": conversation_id,
                },
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )