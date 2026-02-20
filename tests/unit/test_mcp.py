from __future__ import annotations

from agentic_chatbot.mcp import (
    HttpMCPClient,
    MCPConnectorRegistry,
    MCPPromptConnector,
    MCPToolConnector,
)
from agentic_chatbot.schemas import Action, Plan
from agentic_chatbot.skills import JokeSkill


class FakeMCPClient:
    def __init__(self) -> None:
        self.last_tool_call = None
        self.last_prompt_call = None

    def call_tool(self, *, server: str, tool_name: str, arguments: dict) -> str:
        self.last_tool_call = {
            "server": server,
            "tool_name": tool_name,
            "arguments": arguments,
        }
        return "trend=dad_jokes; safety=sfw"

    def get_prompt(self, *, server: str, prompt_name: str, arguments: dict | None = None) -> str:
        self.last_prompt_call = {
            "server": server,
            "prompt_name": prompt_name,
            "arguments": arguments,
        }
        return "You are a deadpan comedian."


class CaptureLLM:
    def __init__(self, output: str) -> None:
        self.output = output
        self.last_messages = None

    def complete(self, messages, *, temperature: float = 0.2) -> str:
        self.last_messages = messages
        return self.output


class FakeTransport:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls = []

    def post_json(self, url: str, payload: dict, *, timeout: float) -> dict:
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        return self.responses.pop(0)


def test_mcp_registry_resolves_tool_and_prompt() -> None:
    client = FakeMCPClient()
    registry = MCPConnectorRegistry()
    registry.register_tool(
        "joke_policy",
        MCPToolConnector(client=client, server="tools-server", tool_name="policy_lookup"),
    )
    registry.register_prompt(
        "joke_system",
        MCPPromptConnector(client=client, server="prompt-server", prompt_name="joke_prompt"),
    )

    tool_value = registry.call_tool("joke_policy", {"topic": "cats"})
    prompt_value = registry.get_prompt("joke_system", {"style": "dry"})

    assert "dad_jokes" in tool_value
    assert "deadpan" in prompt_value
    assert client.last_tool_call["arguments"]["topic"] == "cats"
    assert client.last_prompt_call["arguments"]["style"] == "dry"


def test_joke_skill_uses_mcp_prompt_and_tool_context() -> None:
    client = FakeMCPClient()
    registry = MCPConnectorRegistry()
    registry.register_tool(
        "joke_policy",
        MCPToolConnector(client=client, server="tools-server", tool_name="policy_lookup"),
    )
    registry.register_prompt(
        "joke_system",
        MCPPromptConnector(client=client, server="prompt-server", prompt_name="joke_prompt"),
    )

    llm = CaptureLLM("mock joke")
    skill = JokeSkill(llm=llm, mcp_registry=registry)
    plan = Plan(
        action=Action.JOKE,
        reason="test",
        params={
            "topic": "cats",
            "mcp_prompt": "joke_system",
            "mcp_tool": "joke_policy",
            "tool_args": {"region": "us"},
        },
    )

    response = skill.run(plan=plan, history=[], user_message="tell me a cat joke")

    assert response.content == "mock joke"
    assert llm.last_messages[0].content == "You are a deadpan comedian."
    assert "External tool context" in llm.last_messages[-1].content
    assert "dad_jokes" in llm.last_messages[-1].content
    assert client.last_tool_call["arguments"]["region"] == "us"


def test_http_mcp_client_call_tool_success() -> None:
    transport = FakeTransport([{"result": "tool output"}])
    client = HttpMCPClient(transport=transport, timeout_seconds=5)

    output = client.call_tool(
        server="https://mcp.example.com",
        tool_name="lookup",
        arguments={"q": "pasta"},
    )

    assert output == "tool output"
    assert transport.calls[0]["url"] == "https://mcp.example.com/tools/call"
    assert transport.calls[0]["payload"]["tool_name"] == "lookup"
    assert transport.calls[0]["payload"]["arguments"]["q"] == "pasta"
    assert transport.calls[0]["timeout"] == 5


def test_http_mcp_client_call_tool_error_payload() -> None:
    transport = FakeTransport([{"error": "tool failed"}])
    client = HttpMCPClient(transport=transport)

    try:
        client.call_tool(server="https://mcp.example.com", tool_name="lookup", arguments={})
        raised = False
    except ValueError as exc:
        raised = True
        assert "tool failed" in str(exc)

    assert raised is True
