from __future__ import annotations

import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from insights_generator.config import load_config
from insights_generator.graph import build_graph
from insights_generator.io_utils import load_dataframe_from_upload
from insights_generator.model_router import get_chat_client
from insights_generator.models import ClarifyRequest
from insights_generator.prompting import load_prompt_pack
from insights_generator.session_store import SessionPayload, delete_session, get_session, put_session

load_dotenv()
config = load_config()
chat_client = get_chat_client(config.model)
prompt_pack = load_prompt_pack(config.prompts_path)

app = FastAPI(title="Insights Generator", version="0.2.0")
graph = build_graph(chat_client, prompt_pack)


def _execute_graph(
    session_id: str,
    dataframe,
    user_prompt: str,
    clarification: str = "",
    use_python_repl: bool = False,
    use_mcp: bool = False,
) -> dict[str, Any]:
    initial_state = {
        "session_id": session_id,
        "dataframe": dataframe,
        "user_prompt": user_prompt,
        "clarification": clarification,
        "use_python_repl": use_python_repl,
        "use_mcp": use_mcp,
    }
    return graph.invoke(initial_state)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict[str, Any]:
    return {
        "provider": config.model.provider,
        "model_name": config.model.model_name,
        "temperature": config.model.temperature,
    }


@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    user_prompt: str = Form(default=""),
    use_python_repl: bool = Form(default=False),
    use_mcp: bool = Form(default=False),
) -> dict[str, Any]:
    try:
        dataframe = load_dataframe_from_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    session_id = str(uuid.uuid4())
    result = _execute_graph(
        session_id=session_id,
        dataframe=dataframe,
        user_prompt=user_prompt,
        use_python_repl=use_python_repl,
        use_mcp=use_mcp,
    )

    if result.get("needs_clarification"):
        put_session(
            session_id,
            SessionPayload(
                dataframe=dataframe,
                initial_prompt=user_prompt,
                use_python_repl=use_python_repl,
                use_mcp=use_mcp,
            ),
        )
        return {
            "session_id": session_id,
            "needs_clarification": True,
            "clarification_question": result.get("clarification_question"),
            "intent": result.get("intent", {}),
        }

    return {
        "session_id": session_id,
        "needs_clarification": False,
        "intent": result.get("intent", {}),
        "analytics": result.get("analytics", {}),
        "visualizations": result.get("visualizations", []),
        "insights": result.get("insights", ""),
    }


@app.post("/clarify")
def clarify(request: ClarifyRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    result = _execute_graph(
        session_id=request.session_id,
        dataframe=session.dataframe,
        user_prompt=session.initial_prompt,
        clarification=request.clarification,
        use_python_repl=session.use_python_repl,
        use_mcp=session.use_mcp,
    )

    if result.get("needs_clarification"):
        return {
            "session_id": request.session_id,
            "needs_clarification": True,
            "clarification_question": result.get("clarification_question"),
            "intent": result.get("intent", {}),
        }

    delete_session(request.session_id)
    return {
        "session_id": request.session_id,
        "needs_clarification": False,
        "intent": result.get("intent", {}),
        "analytics": result.get("analytics", {}),
        "visualizations": result.get("visualizations", []),
        "insights": result.get("insights", ""),
    }
