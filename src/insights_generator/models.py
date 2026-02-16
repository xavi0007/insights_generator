from __future__ import annotations

from pydantic import BaseModel


class ClarifyRequest(BaseModel):
    session_id: str
    clarification: str
