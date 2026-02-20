from __future__ import annotations

from types import SimpleNamespace

from agentic_chatbot.factory import ChatbotFactory
from agentic_chatbot.llm import AnthropicChatClient, GoogleChatClient, OpenAIChatClient


class DummyOpenAISDK:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )


class DummyAnthropicSDK:
    def __init__(self) -> None:
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        return SimpleNamespace(content=[SimpleNamespace(text="ok")])


class DummyGoogleSDK:
    def generate_content(self, prompt, generation_config=None):
        return SimpleNamespace(text="ok")


def test_factory_builds_openai_client() -> None:
    factory = ChatbotFactory(
        provider="openai", model="gpt-test", api_key="k", sdk_client=DummyOpenAISDK()
    )

    llm = factory.build_llm()

    assert isinstance(llm, OpenAIChatClient)


def test_factory_builds_anthropic_client() -> None:
    factory = ChatbotFactory(
        provider="anthropic",
        model="claude-test",
        api_key="k",
        sdk_client=DummyAnthropicSDK(),
    )

    llm = factory.build_llm()

    assert isinstance(llm, AnthropicChatClient)


def test_factory_builds_google_client() -> None:
    factory = ChatbotFactory(
        provider="google",
        model="gemini-test",
        api_key="k",
        sdk_client=DummyGoogleSDK(),
    )

    llm = factory.build_llm()

    assert isinstance(llm, GoogleChatClient)


def test_factory_from_env_reads_provider_specific_keys(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-from-env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    factory = ChatbotFactory.from_env()

    assert factory.provider == "anthropic"
    assert factory.model == "claude-from-env"
    assert factory.api_key == "anthropic-key"
