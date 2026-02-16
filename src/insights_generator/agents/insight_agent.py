from __future__ import annotations

from insights_generator.model_router import ChatClient
from insights_generator.state import GraphState


def _top_anomaly_columns(anomaly_summary: dict[str, int], limit: int = 3) -> list[tuple[str, int]]:
    return sorted(anomaly_summary.items(), key=lambda kv: kv[1], reverse=True)[:limit]


def _heuristic_insight(state: GraphState) -> str:
    analytics = state.get("analytics", {})
    anomaly_summary = analytics.get("anomaly_summary", {})
    high_variance = analytics.get("high_variance_columns", [])
    long_tail = analytics.get("long_tail_columns", [])
    row_count = analytics.get("row_count", 0)

    anomaly_top = _top_anomaly_columns(anomaly_summary)

    lines: list[str] = []
    lines.append(f"Dataset contains {row_count} rows and {analytics.get('column_count', 0)} columns.")
    if anomaly_top:
        lines.append(
            "Top anomaly-heavy columns by IQR count: "
            + ", ".join([f"{col} ({count})" for col, count in anomaly_top])
            + "."
        )
    if high_variance:
        lines.append("High variance detected in: " + ", ".join(high_variance) + ".")
    else:
        lines.append("No numeric column crossed the high-variance threshold (CV > 1.0).")
    if long_tail:
        lines.append("Long-tail behavior detected in: " + ", ".join(long_tail) + ".")
    else:
        lines.append("No strong long-tail behavior detected from skew threshold.")
    lines.append("Generated visualizations include trend, distribution, anomaly, and variance charts.")
    return "\n".join(lines)


def build_insight_agent(chat_client: ChatClient):
    def run_insight_agent(state: GraphState) -> GraphState:
        analytics = state.get("analytics", {})
        intent = state.get("intent", {})
        heuristic = _heuristic_insight(state)

        prompt = f"""
You are a senior analytics consultant. Produce concise actionable insights.
Use this intent: {intent}
Use this analytics summary: {analytics}
Available chart artifacts: {state.get('visualizations', [])}

Write 5-8 bullet-style lines including trends, anomalies, variance, long-tail observations, and recommended next action.
""".strip()

        llm_text = chat_client.invoke_text(prompt)
        state["insights"] = llm_text if llm_text else heuristic
        return state

    return run_insight_agent
