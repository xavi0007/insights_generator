from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_name: str
    temperature: float
    openai_api_key: str
    openai_base_url: str
    anthropic_api_key: str


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    prompts_path: str



def load_config() -> AppConfig:
    provider = os.getenv("MODEL_PROVIDER", "none").strip().lower() or "none"
    model_name = os.getenv("MODEL_NAME", "")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
    return AppConfig(
        model=ModelConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        ),
        prompts_path=os.getenv("PROMPTS_PATH", "prompts/insights_prompts.yaml"),
    )
