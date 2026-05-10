from __future__ import annotations

from pathlib import Path


def load_project_env() -> None:
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    if not dotenv_path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: python-dotenv. Install it with `python3 -m pip install -r requirements.txt`."
        ) from exc

    load_dotenv(dotenv_path=dotenv_path, override=False)
