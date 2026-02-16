from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class GraphState(TypedDict, total=False):
    session_id: str
    dataframe: pd.DataFrame
    user_prompt: str
    clarification: str
    needs_clarification: bool
    clarification_question: str
    intent: dict[str, Any]
    analytics: dict[str, Any]
    visualizations: list[dict[str, Any]]
    insights: str
    use_python_repl: bool
    use_mcp: bool
