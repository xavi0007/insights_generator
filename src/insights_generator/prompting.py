from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_prompt_pack(path: str) -> dict[str, Any]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        return {}

    try:
        with prompt_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
