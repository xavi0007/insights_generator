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
- `PROMPTS_PATH`: YAML prompt pack path (default `prompts/insights_prompts.yaml`)

If provider config is missing/unavailable, the app falls back to deterministic heuristics.

## Externalized prompts and few-shots
- Prompt pack file: `prompts/insights_prompts.yaml`
- `intent` section contains parser rules and intent few-shot JSON examples.
- `insight` section contains business logic constraints and insight few-shot examples.
- Update this YAML to tune domain logic without code changes.

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

## Agentic chatbot example (clarify + joke + recipe)
This repo also includes a maintainable/testable agentic chatbot module in `src/agentic_chatbot`:
- `planner.py`: intent routing planner (`clarify`, `joke`, `recipe`)
- `skills.py`: isolated skills (clarification, joke generation, recipe generation)
- `agent.py`: orchestration layer for dispatching plans to skills
- `llm.py`: OpenAI adapter behind an `LLMClient` protocol for dependency injection
- `factory.py`: `ChatbotFactory` for provider swapping (`openai`, `anthropic`, `google`)
- `mcp.py`: MCP connectors/registry for external tool and prompt servers

### Why this is testable
- Planner, agent, and skills use interface-based dependency injection.
- Unit tests use stubs/mocks, so no real OpenAI calls are needed.
- OpenAI payload formatting is verified in isolation.

### Run tests
```bash
pip install -e .[dev]
pytest
```

### Run chatbot CLI
```bash
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-4o-mini
export LLM_PROVIDER=openai
agentic-chatbot
```

Provider-specific env keys:
- OpenAI: `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, `OPENAI_MODEL`
- Anthropic: `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`
- Google: `LLM_PROVIDER=google`, `GOOGLE_API_KEY`, `GOOGLE_MODEL`

### MCP connectors (tools + prompts)
Use `MCPConnectorRegistry` to register aliases for MCP prompt and tool endpoints, then inject it into `ChatbotFactory`.

```python
from agentic_chatbot.factory import ChatbotFactory
from agentic_chatbot.mcp import MCPConnectorRegistry, MCPPromptConnector, MCPToolConnector

registry = MCPConnectorRegistry()
registry.register_prompt(
    "joke_system",
    MCPPromptConnector(client=mcp_client, server="prompt-server", prompt_name="joke_prompt"),
)
registry.register_tool(
    "joke_policy",
    MCPToolConnector(client=mcp_client, server="tools-server", tool_name="policy_lookup"),
)

agent = ChatbotFactory(provider="openai", mcp_registry=registry).build_agent()
```

Planner params can reference connectors:
- `mcp_prompt`: prompt alias (for system prompt override)
- `mcp_tool`: tool alias (tool output is appended as extra context)
- `prompt_args` / `tool_args`: optional argument dictionaries
