from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from insights_generator.agents.analytics_agent import run_analytics_agent
from insights_generator.agents.insight_agent import build_insight_agent
from insights_generator.agents.intent_agent import build_intent_agent
from insights_generator.agents.visualization_agent import run_visualization_agent
from insights_generator.model_router import ChatClient
from insights_generator.state import GraphState


def _route_after_intent(state: GraphState) -> str:
    if state.get("needs_clarification"):
        return "end"
    return "analytics"


def build_graph(chat_client: ChatClient, prompt_pack: dict[str, Any] | None = None):
    prompt_pack = prompt_pack or {}
    graph = StateGraph(GraphState)

    graph.add_node("intent", build_intent_agent(chat_client, prompt_pack.get("intent", {})))
    graph.add_node("analytics", run_analytics_agent)
    graph.add_node("visualization", run_visualization_agent)
    graph.add_node("insight", build_insight_agent(chat_client, prompt_pack.get("insight", {})))

    graph.add_edge(START, "intent")
    graph.add_conditional_edges("intent", _route_after_intent, {"analytics": "analytics", "end": END})
    graph.add_edge("analytics", "visualization")
    graph.add_edge("visualization", "insight")
    graph.add_edge("insight", END)

    return graph.compile()
