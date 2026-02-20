from __future__ import annotations

from agentic_chatbot.planner import Planner
from agentic_chatbot.schemas import Action, ChatMessage, Role


class StubLLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def complete(self, messages, *, temperature: float = 0.2) -> str:
        return self.outputs.pop(0)


def test_planner_routes_to_joke() -> None:
    llm = StubLLM([
        '{"action":"joke","reason":"user asked for joke","params":{"topic":"cats"},"clarifying_question":null}'
    ])
    planner = Planner(llm=llm)

    plan = planner.plan(history=[], user_message="Tell me a cat joke")

    assert plan.action == Action.JOKE
    assert plan.params["topic"] == "cats"


def test_planner_falls_back_to_clarify_on_invalid_json() -> None:
    llm = StubLLM(["not-json"])
    planner = Planner(llm=llm)

    plan = planner.plan(history=[], user_message="help")

    assert plan.action == Action.CLARIFY
    assert "clarify" in (plan.clarifying_question or "").lower()


def test_planner_extracts_json_from_wrapped_text() -> None:
    llm = StubLLM(
        [
            'Here you go:\n{"action":"recipe","reason":"food request","params":{"ingredients":"eggs"},"clarifying_question":null}'
        ]
    )
    planner = Planner(llm=llm)

    plan = planner.plan(
        history=[ChatMessage(role=Role.USER, content="I want breakfast")],
        user_message="Can you suggest a recipe?",
    )

    assert plan.action == Action.RECIPE
    assert plan.params["ingredients"] == "eggs"
