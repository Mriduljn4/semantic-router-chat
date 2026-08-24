from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent import run_agent as execute_agent
from src.rewriter import rewrite_query as normalize_query
from src.router import route as decide_route
from src.tracing import configure_tracing


class QueryState(TypedDict):
    query: str
    rewritten_query: str
    routed_agent: str
    router_scores: dict[str, float]
    answer: str
    llm_provider_used: str


def route(state: QueryState) -> dict:
    decision = decide_route(state["rewritten_query"])
    return {"routed_agent": decision.routed_agent, "router_scores": decision.router_scores}


def run_agent(state: QueryState) -> dict:
    result = execute_agent(state["routed_agent"], state["rewritten_query"])
    return {"answer": result.answer, "llm_provider_used": result.provider_used}


def rewrite_query(state: QueryState) -> dict:
    return {"rewritten_query": normalize_query(state["query"])}


configure_tracing()
_builder = StateGraph(QueryState)
_builder.add_node("rewrite_query", rewrite_query)
_builder.add_node("route", route)
_builder.add_node("run_agent", run_agent)
_builder.add_edge(START, "rewrite_query")
_builder.add_edge("rewrite_query", "route")
_builder.add_edge("route", "run_agent")
_builder.add_edge("run_agent", END)
graph = _builder.compile()


def run_query(query: str) -> dict:
    state = graph.invoke({"query": query})
    return {
        key: state[key]
        for key in ("answer", "routed_agent", "router_scores", "llm_provider_used", "rewritten_query")
    }
