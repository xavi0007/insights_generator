from __future__ import annotations

from agentic_chatbot.agent import AgenticChatbot
from agentic_chatbot.planner import Planner
from agentic_chatbot.schemas import Action
from agentic_chatbot.skills import ClarifySkill, JokeSkill, RecipeSkill


class StubLLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def complete(self, messages, *, temperature: float = 0.2) -> str:
        return self.outputs.pop(0)


def build_agent(outputs: list[str]) -> AgenticChatbot:
    llm = StubLLM(outputs)
    return AgenticChatbot(
        planner=Planner(llm=llm),
        clarify_skill=ClarifySkill(),
        joke_skill=JokeSkill(llm=llm),
        recipe_skill=RecipeSkill(llm=llm),
    )


def test_agent_dispatches_to_joke_skill() -> None:
    agent = build_agent(
        [
            '{"action":"joke","reason":"asked joke","params":{"topic":"dogs"},"clarifying_question":null}',
            "Why did the dog sit in the shade? Because he did not want to be a hot dog.",
        ]
    )

    response = agent.respond(history=[], user_message="Tell me a joke")

    assert response.action == Action.JOKE
    assert "dog" in response.content.lower()


def test_agent_dispatches_to_recipe_skill() -> None:
    agent = build_agent(
        [
            '{"action":"recipe","reason":"asked recipe","params":{"ingredients":"tomato, pasta","servings":2},"clarifying_question":null}',
            "Tomato Pasta\nIngredients: tomato, pasta\nSteps: 1) Boil pasta 2) Add tomato sauce",
        ]
    )

    response = agent.respond(history=[], user_message="Need a quick pasta recipe")

    assert response.action == Action.RECIPE
    assert "ingredients" in response.content.lower()


def test_agent_clarifies_when_planner_asks() -> None:
    agent = build_agent(
        [
            '{"action":"clarify","reason":"ambiguous","params":{},"clarifying_question":"Do you want a joke or a recipe?"}'
        ]
    )

    response = agent.respond(history=[], user_message="Do something fun")

    assert response.action == Action.CLARIFY
    assert response.content == "Do you want a joke or a recipe?"
