import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent import run_agent as execute_agent
from src.router import route as decide_route
from src.tracing import configure_tracing


class QueryState(TypedDict):
    """State passed between the deterministic router and specialist graph nodes."""

    query: str
    routed_agent: str
    router_scores: dict[str, float]
    intent_classifier: str
    answer: str
    llm_provider_used: str
    tools_used: list[str]


async def route(state: QueryState) -> dict:
    """Populate the chosen specialist and the intent-classifier similarity scores."""
    # Chroma's Python client is synchronous, so run it outside the ASGI event loop.
    decision = await asyncio.to_thread(decide_route, state["query"])
    return {
        "routed_agent": decision.routed_agent,
        "router_scores": decision.router_scores,
        "intent_classifier": decision.classifier_used,
    }


async def run_agent(state: QueryState) -> dict:
    """Run the selected LangChain specialist with the original guarded query."""
    # The specialist currently uses synchronous provider clients; isolate that work
    # so other API requests can continue while an LLM response is generated.
    result = await asyncio.to_thread(execute_agent, state["routed_agent"], state["query"])
    return {
        "answer": result.answer,
        "llm_provider_used": result.provider_used,
        "tools_used": result.tools_used,
    }


# Configure LangSmith before compiling so graph and model runs are traceable.
configure_tracing()
_builder = StateGraph(QueryState)
_builder.add_node("route", route)
_builder.add_node("run_agent", run_agent)
_builder.add_edge(START, "route")
_builder.add_edge("route", "run_agent")
_builder.add_edge("run_agent", END)
graph = _builder.compile()


def run_query(query: str) -> dict:
    """Execute the route → specialist LangGraph workflow and return API fields."""
    state = graph.invoke({"query": query})
    return {
        key: state[key]
        for key in ("answer", "routed_agent", "router_scores", "intent_classifier", "llm_provider_used", "tools_used")
    }


async def run_query_async(query: str) -> dict:
    """Asynchronously execute the graph for use by FastAPI request handlers."""
    state = await graph.ainvoke({"query": query})
    return {
        key: state[key]
        for key in ("answer", "routed_agent", "router_scores", "intent_classifier", "llm_provider_used", "tools_used")
    }
