from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from insights_generator.config import ModelConfig


class ChatClient(Protocol):
    def invoke_text(self, prompt: str) -> str:
        ...


@dataclass
class HeuristicClient:
    def invoke_text(self, prompt: str) -> str:
        return ""


@dataclass
class OpenAIClient:
    model_name: str
    temperature: float
    api_key: str
    base_url: str

    def __post_init__(self) -> None:
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._llm = ChatOpenAI(**kwargs)

    def invoke_text(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        return str(getattr(response, "content", "")).strip()


@dataclass
class AnthropicClient:
    model_name: str
    temperature: float
    api_key: str

    def __post_init__(self) -> None:
        from langchain_anthropic import ChatAnthropic

        self._llm = ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
        )

    def invoke_text(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        return str(getattr(response, "content", "")).strip()



def get_chat_client(config: ModelConfig) -> ChatClient:
    provider = config.provider

    if provider == "openai" and config.openai_api_key and config.model_name:
        try:
            return OpenAIClient(
                model_name=config.model_name,
                temperature=config.temperature,
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
            )
        except Exception:
            return HeuristicClient()

    if provider == "anthropic" and config.anthropic_api_key and config.model_name:
        try:
            return AnthropicClient(
                model_name=config.model_name,
                temperature=config.temperature,
                api_key=config.anthropic_api_key,
            )
        except Exception:
            return HeuristicClient()

    return HeuristicClient()
