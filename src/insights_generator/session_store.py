from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SessionPayload:
    dataframe: pd.DataFrame
    initial_prompt: str
    use_python_repl: bool = False
    use_mcp: bool = False


SESSION_STORE: dict[str, SessionPayload] = {}


def put_session(session_id: str, payload: SessionPayload) -> None:
    SESSION_STORE[session_id] = payload


def get_session(session_id: str) -> SessionPayload | None:
    return SESSION_STORE.get(session_id)


def delete_session(session_id: str) -> None:
    SESSION_STORE.pop(session_id, None)
