from __future__ import annotations

from types import SimpleNamespace

from agentic_chatbot.llm import AnthropicChatClient, GoogleChatClient, OpenAIChatClient
from agentic_chatbot.schemas import ChatMessage, Role


class DummyOpenAISDK:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_create_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_create_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response_text))]
        )


class DummyAnthropicSDK:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_create_kwargs = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.last_create_kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text=self.response_text)])


class DummyGoogleSDK:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_prompt = None
        self.last_config = None

    def generate_content(self, prompt, generation_config=None):
        self.last_prompt = prompt
        self.last_config = generation_config
        return SimpleNamespace(text=self.response_text)


def test_openai_client_sends_expected_payload() -> None:
    sdk = DummyOpenAISDK("hello")
    client = OpenAIChatClient(model="gpt-test", sdk_client=sdk)

    content = client.complete(
        [
            ChatMessage(role=Role.SYSTEM, content="sys"),
            ChatMessage(role=Role.USER, content="hi"),
        ],
        temperature=0.3,
    )

    assert content == "hello"
    assert sdk.last_create_kwargs["model"] == "gpt-test"
    assert sdk.last_create_kwargs["temperature"] == 0.3
    assert sdk.last_create_kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_openai_client_raises_on_empty_content() -> None:
    sdk = DummyOpenAISDK("")
    client = OpenAIChatClient(model="gpt-test", sdk_client=sdk)

    try:
        client.complete([ChatMessage(role=Role.USER, content="hi")])
        raised = False
    except ValueError:
        raised = True

    assert raised is True


def test_anthropic_client_uses_messages_api() -> None:
    sdk = DummyAnthropicSDK("claude output")
    client = AnthropicChatClient(model="claude-test", sdk_client=sdk)

    content = client.complete(
        [
            ChatMessage(role=Role.SYSTEM, content="sys"),
            ChatMessage(role=Role.USER, content="hello"),
        ],
        temperature=0.5,
    )

    assert content == "claude output"
    assert sdk.last_create_kwargs["model"] == "claude-test"
    assert sdk.last_create_kwargs["temperature"] == 0.5


def test_google_client_uses_generate_content() -> None:
    sdk = DummyGoogleSDK("gemini output")
    client = GoogleChatClient(model="gemini-test", sdk_client=sdk)

    content = client.complete([ChatMessage(role=Role.USER, content="hello")], temperature=0.6)

    assert content == "gemini output"
    assert "user: hello" in sdk.last_prompt
    assert sdk.last_config == {"temperature": 0.6}
