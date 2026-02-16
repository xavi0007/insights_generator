from __future__ import annotations

import json

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


def _build_insight_prompt(state: GraphState, prompt_cfg: dict) -> str:
    system_instructions = prompt_cfg.get(
        "system_instructions",
        "You are a senior analytics consultant writing business-facing insights.",
    )
    business_logic = prompt_cfg.get("business_logic", [])
    output_instructions = prompt_cfg.get(
        "output_instructions",
        "Write concise actionable insights with trend, anomalies, variance, long-tail, and next action.",
    )
    few_shots = prompt_cfg.get("few_shots", [])

    lines: list[str] = [str(system_instructions).strip()]
    if isinstance(business_logic, list) and business_logic:
        lines.append("Business logic constraints:")
        for rule in business_logic:
            lines.append(f"- {rule}")

    if output_instructions:
        lines.append("Output instructions:")
        lines.append(str(output_instructions).strip())

    if isinstance(few_shots, list) and few_shots:
        lines.append("Few-shot examples:")
        for ex in few_shots:
            input_obj = ex.get("input", {})
            assistant = ex.get("assistant", "")
            if input_obj and assistant:
                lines.append(f"Input: {json.dumps(input_obj)}")
                lines.append(f"Assistant: {str(assistant).strip()}")

    lines.append(f"Intent: {state.get('intent', {})}")
    lines.append(f"Analytics summary: {state.get('analytics', {})}")
    lines.append(f"Available chart artifacts: {state.get('visualizations', [])}")
    return "\n".join(lines).strip()


def build_insight_agent(chat_client: ChatClient, prompt_cfg: dict):
    def run_insight_agent(state: GraphState) -> GraphState:
        heuristic = _heuristic_insight(state)
        prompt = _build_insight_prompt(state, prompt_cfg)

        llm_text = chat_client.invoke_text(prompt)
        state["insights"] = llm_text if llm_text else heuristic
        return state

    return run_insight_agent
