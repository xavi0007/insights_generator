from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from insights_generator.state import GraphState


IQR_MULTIPLIER = 1.5
LONG_TAIL_SKEW_THRESHOLD = 1.0
HIGH_VARIANCE_CV_THRESHOLD = 1.0


def _safe_mode(series: pd.Series) -> list[Any]:
    modes = series.mode(dropna=True)
    return modes.head(3).tolist() if not modes.empty else []


def _numeric_column_analytics(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return {}

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    outliers = series[(series < lower) | (series > upper)]

    mean_val = float(series.mean())
    std_val = float(series.std(ddof=0))
    cv = std_val / mean_val if mean_val else float("inf")

    return {
        "count": int(series.count()),
        "mean": mean_val,
        "average": mean_val,
        "median": float(series.median()),
        "mode": _safe_mode(series),
        "std": std_val,
        "variance": float(series.var(ddof=0)),
        "cv": cv,
        "high_variance": bool(cv > HIGH_VARIANCE_CV_THRESHOLD),
        "skew": float(series.skew()),
        "long_tail_detected": bool(abs(float(series.skew())) > LONG_TAIL_SKEW_THRESHOLD),
        "iqr": iqr,
        "iqr_bounds": {"lower": lower, "upper": upper},
        "anomaly_count": int(outliers.count()),
        "anomaly_rate": float(outliers.count() / max(series.count(), 1)),
        "anomaly_examples": outliers.head(10).tolist(),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def run_analytics_agent(state: GraphState) -> GraphState:
    df = state["dataframe"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_analytics: dict[str, Any] = {
        col: _numeric_column_analytics(df, col)
        for col in numeric_cols
    }

    high_variance_columns = [
        col for col, info in numeric_analytics.items() if info and info.get("high_variance")
    ]
    long_tail_columns = [
        col for col, info in numeric_analytics.items() if info and info.get("long_tail_detected")
    ]
    anomaly_summary = {
        col: info.get("anomaly_count", 0)
        for col, info in numeric_analytics.items()
    }

    state["analytics"] = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_analytics": numeric_analytics,
        "high_variance_columns": high_variance_columns,
        "long_tail_columns": long_tail_columns,
        "anomaly_summary": anomaly_summary,
    }
    return state
