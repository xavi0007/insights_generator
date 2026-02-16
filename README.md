# Insights Generator (LangGraph Multi-Agent)

Multi-agent analytics pipeline for CSV/Parquet data with configurable LLM backend.

## Features
- Upload CSV/Parquet data.
- Intent recognition with clarification loop.
- Data analytics agent computes:
  - mean, median, mode, average
  - anomaly detection using IQR
  - long-tail detection (skew)
  - high-variance detection (coefficient of variation)
- Visualization agent generates Plotly charts from templates.
- Insight agent writes trend/findings summary.
- Swappable model backend via environment variables.

## Model swapping
Set in `.env`:
- `MODEL_PROVIDER`: `openai`, `anthropic`, or `none`
- `MODEL_NAME`: model id for the provider

If provider config is missing/unavailable, the app falls back to deterministic heuristics.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[providers]
uvicorn insights_generator.api:app --reload
```

Open docs: `http://127.0.0.1:8000/docs`

## Endpoints
- `POST /analyze` (multipart form)
  - `file`: CSV or Parquet
  - `user_prompt`: optional
  - `use_python_repl`: optional bool
  - `use_mcp`: optional bool
- `POST /clarify` (JSON)
  - `session_id`
  - `clarification`

Generated charts are saved in `artifacts/<session_id>/`.
