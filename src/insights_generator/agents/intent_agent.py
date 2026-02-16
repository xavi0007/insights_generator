from __future__ import annotations

import json
import re
from typing import Any

from insights_generator.model_router import ChatClient
from insights_generator.state import GraphState


EXPECTED_SCHEMA = {
    "requested_focus": ["summary"],
    "visualization_preferences": [],
    "needs_clarification": False,
    "clarification_question": "",
}


def _infer_visualization_preferences(text: str) -> list[str]:
    lowered = text.lower()
    prefs = []
    keyword_map = {
        "line": ["trend", "time", "line"],
        "bar": ["bar", "compare", "comparison"],
        "histogram": ["distribution", "hist", "spread", "long tail"],
        "scatter": ["scatter", "outlier", "anomaly"],
        "box": ["variance", "box", "dispersion"],
    }
    for chart, words in keyword_map.items():
        if any(word in lowered for word in words):
            prefs.append(chart)
    return list(dict.fromkeys(prefs))


def _heuristic_intent(combined: str) -> dict[str, Any]:
    if not combined:
        return {
            **EXPECTED_SCHEMA,
            "needs_clarification": True,
            "clarification_question": (
                "What insight do you want first: trend analysis, anomaly detection, "
                "variance/dispersion, distribution/long-tail, or comparison?"
            ),
        }

    visualization_preferences = _infer_visualization_preferences(combined)
    lowered = combined.lower()

    requested_focus = []
    focus_map = {
        "trend": ["trend", "over time", "trajectory", "growth", "decline"],
        "anomaly": ["anomaly", "outlier", "unusual"],
        "variance": ["variance", "volatile", "dispersion", "stability"],
        "distribution": ["distribution", "long tail", "tail", "skew"],
        "summary": ["summary", "overview", "kpi", "basic stats", "statistics"],
    }
    for label, keys in focus_map.items():
        if any(key in lowered for key in keys):
            requested_focus.append(label)

    vague_inputs = {"analyze", "analysis", "insights", "show insights", "visualize"}
    needs_clarification = lowered in vague_inputs
    if not requested_focus and not visualization_preferences and len(combined.split()) < 4:
        needs_clarification = True

    return {
        "requested_focus": requested_focus or ["summary"],
        "visualization_preferences": visualization_preferences,
        "needs_clarification": needs_clarification,
        "clarification_question": (
            "Please choose your first priority: trend, anomalies, long-tail/distribution, high variance, or executive summary."
            if needs_clarification
            else ""
        ),
    }


def _llm_intent(client: ChatClient, combined: str) -> dict[str, Any]:
    prompt = f"""
You are an intent parser for analytics requests.
Return strict JSON with keys:
requested_focus (list[str]), visualization_preferences (list[str]), needs_clarification (bool), clarification_question (str).
Valid focus labels: trend, anomaly, variance, distribution, summary.
Valid visualization preferences: line, bar, histogram, scatter, box.

User request: {combined!r}
""".strip()
    output = client.invoke_text(prompt)
    try:
        parsed = json.loads(output)
        return {
            "requested_focus": parsed.get("requested_focus") or ["summary"],
            "visualization_preferences": parsed.get("visualization_preferences") or [],
            "needs_clarification": bool(parsed.get("needs_clarification", False)),
            "clarification_question": parsed.get("clarification_question", ""),
        }
    except Exception:
        return _heuristic_intent(combined)


def build_intent_agent(chat_client: ChatClient):
    def run_intent_agent(state: GraphState) -> GraphState:
        prompt = (state.get("user_prompt") or "").strip()
        clarification = (state.get("clarification") or "").strip()
        combined = f"{prompt} {clarification}".strip()

        parsed = _llm_intent(chat_client, combined) if combined else _heuristic_intent(combined)
        column_hints = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", combined)

        state["needs_clarification"] = parsed["needs_clarification"]
        state["clarification_question"] = parsed["clarification_question"]
        state["intent"] = {
            "raw_request": combined,
            "requested_focus": parsed["requested_focus"],
            "visualization_preferences": parsed["visualization_preferences"],
            "column_hints": column_hints,
        }
        return state

    return run_intent_agent
