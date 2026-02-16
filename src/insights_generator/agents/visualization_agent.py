from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.express as px

from insights_generator.state import GraphState
from insights_generator.templates.chart_templates import CHART_TEMPLATES


def _write_figure(fig, out_dir: Path, name: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{name}.html"
    json_path = out_dir / f"{name}.json"
    fig.write_html(html_path)
    json_path.write_text(fig.to_json(), encoding="utf-8")
    return {
        "name": name,
        "html_path": str(html_path),
        "json_path": str(json_path),
        "template": CHART_TEMPLATES.get(name, {}),
    }


def _try_python_repl_plotly(state: GraphState) -> None:
    if not state.get("use_python_repl"):
        return
    try:
        from langchain_experimental.utilities import PythonREPL  # type: ignore
    except Exception:
        return

    repl = PythonREPL()
    repl.run("ready = True")


def run_visualization_agent(state: GraphState) -> GraphState:
    df = state["dataframe"]
    analytics = state.get("analytics", {})
    numeric_cols = analytics.get("numeric_columns", [])
    out_dir = Path("artifacts") / state["session_id"]

    _try_python_repl_plotly(state)

    visualizations: list[dict[str, Any]] = []
    if not numeric_cols:
        state["visualizations"] = visualizations
        return state

    primary_numeric = numeric_cols[0]

    hist = px.histogram(df, x=primary_numeric, nbins=50, title=f"Distribution of {primary_numeric}")
    visualizations.append(_write_figure(hist, out_dir, "distribution"))

    scatter = px.scatter(df.reset_index(), x="index", y=primary_numeric, title=f"Anomaly View for {primary_numeric}")
    visualizations.append(_write_figure(scatter, out_dir, "anomaly"))

    melted = df[numeric_cols].melt(var_name="metric", value_name="value")
    box = px.box(melted, x="metric", y="value", title="Variance Overview")
    visualizations.append(_write_figure(box, out_dir, "variance"))

    line = px.line(df.reset_index(), x="index", y=primary_numeric, title=f"Trend of {primary_numeric}")
    visualizations.append(_write_figure(line, out_dir, "trend"))

    if state.get("use_mcp"):
        visualizations.append(
            {
                "name": "mcp_hook",
                "template": {"description": "MCP execution hook requested."},
                "meta": {"status": "ready_for_mcp_chart_execution"},
            }
        )

    state["visualizations"] = visualizations
    return state
