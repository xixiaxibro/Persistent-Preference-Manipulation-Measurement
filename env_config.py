from __future__ import annotations

from pathlib import Path


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: python-dotenv. Install it with `python3 -m pip install -r requirements.txt`."
        ) from exc

    dotenv_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)
